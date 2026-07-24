//===--- OrcIncrementalExecutor.cpp - Orc Incremental Execution -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an Orc-based incremental code execution.
//
//===----------------------------------------------------------------------===//

#include "OrcIncrementalExecutor.h"
#include "clang/Interpreter/PartialTranslationUnit.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/Orc/DylibManager.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

// Force linking some of the runtimes that helps attaching to a debugger.
LLVM_ATTRIBUTE_USED void linkComponents() {
  llvm::errs() << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}

namespace clang {
OrcIncrementalExecutor::OrcIncrementalExecutor(
    llvm::orc::ThreadSafeContext &TSC)
    : TSCtx(TSC) {}

OrcIncrementalExecutor::OrcIncrementalExecutor(
    llvm::orc::ThreadSafeContext &TSC, llvm::orc::LLJITBuilder &JITBuilder,
    llvm::Error &Err)
    : TSCtx(TSC) {
  using namespace llvm::orc;
  llvm::ErrorAsOutParameter EAO(&Err);

  if (auto JitOrErr = JITBuilder.create())
    Jit = std::move(*JitOrErr);
  else {
    Err = JitOrErr.takeError();
    return;
  }
}

OrcIncrementalExecutor::~OrcIncrementalExecutor() {}

// Bind weak globals the executor process already defines.
//
// A singleton *defined* in a header -- a function-local static in an inline
// function, or a C++17 inline variable -- compiles to a weak/linkonce_odr
// global variable. The JIT materializes every definition it is handed, and
// the process-symbol generators are only consulted for symbols the JITDylib
// lacks, so interpreter code that repeats such a definition silently gets a
// second instance of state the process already owns: two copies of one
// singleton, initialized and destroyed independently, where a dynamic linker
// would have bound both references to one copy.
//
// Demote those definitions to external declarations before the module
// reaches the JIT, so they resolve to the process copy. Only mutable
// variables move: duplicated constants are harmless under the ODR (and
// folding them keeps JITed code fast), functions may legitimately be
// re-JITed, and thread_locals are left to the JIT's TLS machinery. A
// dynamically initialized static moves together with its _ZGV guard or not
// at all -- sharing the data but not the guard would re-run initialization
// on the process copy, and the reverse would leave the JITed copy
// uninitialized. Variables on the llvm.used/llvm.compiler.used lists are
// left alone. ELF only for now: COFF does not export weak definitions for
// this to bind against, and Mach-O needs separate validation.
llvm::Error OrcIncrementalExecutor::bindProcessWeakGlobals(llvm::Module &M) {
  if (!M.getTargetTriple().isOSBinFormatELF())
    return llvm::Error::success();

  llvm::SmallPtrSet<const llvm::GlobalValue *, 8> Used;
  {
    llvm::SmallVector<llvm::GlobalValue *, 8> UsedVec;
    llvm::collectUsedGlobalVariables(M, UsedVec, /*CompilerUsed=*/false);
    Used.insert_range(UsedVec);
    UsedVec.clear();
    llvm::collectUsedGlobalVariables(M, UsedVec, /*CompilerUsed=*/true);
    Used.insert_range(UsedVec);
  }

  llvm::SmallVector<std::pair<llvm::GlobalVariable *, llvm::GlobalVariable *>>
      Candidates;
  for (llvm::GlobalVariable &GV : M.globals()) {
    if (GV.isDeclaration() || !GV.isWeakForLinker() || GV.isThreadLocal() ||
        GV.isConstant() || Used.contains(&GV))
      continue;

    llvm::StringRef Name = GV.getName();
    // Guards are only ever demoted together with their variable, below.
    if (Name.starts_with("_ZGV"))
      continue;

    llvm::GlobalVariable *Guard = nullptr;
    if (Name.starts_with("_Z")) {
      Guard = M.getNamedGlobal(("_ZGV" + Name.drop_front(2)).str());
      if (Guard && Guard->isDeclaration())
        Guard = nullptr;
      if (Guard && Used.contains(Guard))
        continue; // the pair cannot move
    }
    Candidates.push_back({&GV, Guard});
  }
  if (Candidates.empty())
    return llvm::Error::success();

  // One handle for the executor process's global namespace, valid for both
  // in-process and out-of-process executors (the lookup happens executor
  // side either way).
  if (!ProcessDylibHandle) {
    auto H = Jit->getDylibMgr().loadDylib(nullptr);
    if (!H)
      return H.takeError();
    ProcessDylibHandle = *H;
  }

  // ELF has an empty global prefix, so IR names are already linker names.
  auto &ES = Jit->getExecutionSession();
  llvm::orc::SymbolLookupSet Set;
  for (auto &[GV, Guard] : Candidates) {
    Set.add(ES.intern(GV->getName()),
            llvm::orc::SymbolLookupFlags::WeaklyReferencedSymbol);
    if (Guard)
      Set.add(ES.intern(Guard->getName()),
              llvm::orc::SymbolLookupFlags::WeaklyReferencedSymbol);
  }
  auto Result = Jit->getDylibMgr().lookupSymbols(*ProcessDylibHandle, Set);
  if (!Result)
    return Result.takeError();
  llvm::ArrayRef<std::optional<llvm::orc::ExecutorAddr>> Addrs = *Result;

  auto Demote = [](llvm::GlobalVariable &GV) {
    GV.setInitializer(nullptr);
    GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
    GV.setComdat(nullptr);
    GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
    GV.setDSOLocal(false);
  };

  size_t I = 0;
  for (auto &[GV, Guard] : Candidates) {
    std::optional<llvm::orc::ExecutorAddr> VarAddr = Addrs[I++];
    std::optional<llvm::orc::ExecutorAddr> GuardAddr;
    if (Guard)
      GuardAddr = Addrs[I++];
    if (!VarAddr || !*VarAddr || (Guard && (!GuardAddr || !*GuardAddr)))
      continue;
    Demote(*GV);
    if (Guard)
      Demote(*Guard);
  }
  return llvm::Error::success();
}

llvm::Error OrcIncrementalExecutor::addModule(PartialTranslationUnit &PTU) {
  if (llvm::Error Err = bindProcessWeakGlobals(*PTU.TheModule))
    return Err;

  llvm::orc::ResourceTrackerSP RT =
      Jit->getMainJITDylib().createResourceTracker();
  ResourceTrackers[&PTU] = RT;

  return Jit->addIRModule(RT, {std::move(PTU.TheModule), TSCtx});
}

llvm::Error OrcIncrementalExecutor::removeModule(PartialTranslationUnit &PTU) {

  llvm::orc::ResourceTrackerSP RT = std::move(ResourceTrackers[&PTU]);
  if (!RT)
    return llvm::Error::success();

  ResourceTrackers.erase(&PTU);
  if (llvm::Error Err = RT->remove())
    return Err;
  return llvm::Error::success();
}

// Clean up the JIT instance.
llvm::Error OrcIncrementalExecutor::cleanUp() {
  // This calls the global dtors of registered modules.
  return Jit->deinitialize(Jit->getMainJITDylib());
}

llvm::Error OrcIncrementalExecutor::runCtors() const {
  return Jit->initialize(Jit->getMainJITDylib());
}

llvm::Expected<llvm::orc::ExecutorAddr>
OrcIncrementalExecutor::getSymbolAddress(llvm::StringRef Name,
                                         SymbolNameKind NameKind) const {
  using namespace llvm::orc;
  auto SO = makeJITDylibSearchOrder({&Jit->getMainJITDylib(),
                                     Jit->getPlatformJITDylib().get(),
                                     Jit->getProcessSymbolsJITDylib().get()});

  ExecutionSession &ES = Jit->getExecutionSession();

  auto SymOrErr = ES.lookup(SO, (NameKind == SymbolNameKind::LinkerName)
                                    ? ES.intern(Name)
                                    : Jit->mangleAndIntern(Name));
  if (auto Err = SymOrErr.takeError())
    return std::move(Err);
  return SymOrErr->getAddress();
}

llvm::Error OrcIncrementalExecutor::LoadDynamicLibrary(const char *name) {
  // FIXME: Eventually we should put each library in its own JITDylib and
  //        turn off process symbols by default.
  llvm::orc::ExecutionSession &ES = Jit->getExecutionSession();
  auto DLSGOrErr = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
      ES, Jit->getDylibMgr(), name);
  if (!DLSGOrErr)
    return DLSGOrErr.takeError();

  Jit->getProcessSymbolsJITDylib()->addGenerator(std::move(*DLSGOrErr));

  return llvm::Error::success();
}

} // namespace clang
