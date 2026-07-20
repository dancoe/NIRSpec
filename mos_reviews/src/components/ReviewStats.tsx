/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ProgramReview } from '../types';
import { parseSheetDate, getDaysDifference } from '../utils/dateHelpers';

interface ReviewStatsProps {
  programs: ProgramReview[];
  referenceDate: Date;
  activeStatFilter: string | null;
  onStatFilterChange: (filter: string | null) => void;
}

export default function ReviewStats({
  programs,
  referenceDate,
  activeStatFilter,
  onStatFilterChange,
}: ReviewStatsProps) {
  const totalObservations = programs.length;
  
  // Count unique program IDs (filter out falsy inputs)
  const uniquePrograms = new Set(
    programs.map((p) => p.program).filter((val) => typeof val === 'string' && val.trim() !== '')
  ).size;

  let pendingSubmission = 0;
  let overdueSubmission = 0;

  let pendingReview = 0;
  
  let pendingApproval = 0;
  
  let approved = 0;

  programs.forEach((p) => {
    if (p.isCompleted) {
      approved++;
      return;
    }

    // Step 1: Pending Submission
    if (!p.submittedDate || p.submittedDate.trim() === '') {
      pendingSubmission++;
      const subDate = parseSheetDate(p.submissionDueDate);
      if (subDate && getDaysDifference(referenceDate, subDate) < 0) {
        overdueSubmission++;
      }
    } 
    // Step 2: Pending Review (Submitted but not reviewed)
    else if (!p.reviewedDate || p.reviewedDate.trim() === '') {
      pendingReview++;
    } 
    // Step 3: Pending Approval (Reviewed but not fully approved/completed)
    else {
      pendingApproval++;
    }
  });

  const handleCardClick = (filterType: string) => {
    if (activeStatFilter === filterType) {
      onStatFilterChange(null);
    } else {
      onStatFilterChange(filterType);
    }
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-6 gap-3 sm:gap-4 mb-4">
      {/* 1. Programs */}
      <button
        type="button"
        id="stat-programs"
        onClick={() => handleCardClick('programs')}
        className={`border-l-4 border-blue-500 bg-blue-50/50 px-3 py-2 rounded-r shadow-2xs text-left cursor-pointer hover:-translate-y-0.5 hover:shadow-sm transition-all ${
          activeStatFilter === 'programs' ? 'ring-2 ring-blue-500 ring-offset-1 bg-blue-100/50' : ''
        }`}
      >
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl sm:text-2xl font-black text-blue-900">{uniquePrograms}</p>
          <span className="text-[9px] text-blue-500 font-bold whitespace-nowrap">Unique IDs</span>
        </div>
        <p className="text-[10px] uppercase tracking-wider font-extrabold text-blue-600 leading-tight mt-0.5">Programs</p>
      </button>

      {/* 2. Observations */}
      <button
        type="button"
        id="stat-observations"
        onClick={() => handleCardClick('observations')}
        className={`border-l-4 border-slate-400 bg-slate-50/70 px-3 py-2 rounded-r shadow-2xs text-left cursor-pointer hover:-translate-y-0.5 hover:shadow-sm transition-all ${
          activeStatFilter === 'observations' ? 'ring-2 ring-slate-500 ring-offset-1 bg-slate-100/70' : ''
        }`}
      >
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl sm:text-2xl font-black text-slate-800">{totalObservations}</p>
          <span className="text-[9px] text-slate-500 font-bold whitespace-nowrap">Total rows</span>
        </div>
        <p className="text-[10px] uppercase tracking-wider font-extrabold text-slate-500 leading-tight mt-0.5">Observations</p>
      </button>

      {/* 3. Pending Submission */}
      <button
        type="button"
        id="stat-pending-submission"
        onClick={() => handleCardClick('pending_submission')}
        className={`border-l-4 px-3 py-2 rounded-r shadow-2xs text-left cursor-pointer hover:-translate-y-0.5 hover:shadow-sm transition-all ${
          overdueSubmission > 0 ? 'border-rose-500 bg-rose-50/50' : 'border-amber-500 bg-amber-50/60'
        } ${
          activeStatFilter === 'pending_submission'
            ? overdueSubmission > 0
              ? 'ring-2 ring-rose-500 ring-offset-1 bg-rose-100/50'
              : 'ring-2 ring-amber-500 ring-offset-1 bg-amber-100/60'
            : ''
        }`}
      >
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl sm:text-2xl font-black text-amber-905">{pendingSubmission}</p>
          {overdueSubmission > 0 && (
            <span className="text-[9px] text-rose-600 font-extrabold animate-pulse whitespace-nowrap">
              ({overdueSubmission} overdue)
            </span>
          )}
        </div>
        <p className="text-[10px] uppercase tracking-wider font-extrabold text-amber-700 leading-tight mt-0.5">Pending Submission</p>
      </button>

      {/* 4. Pending Review */}
      <button
        type="button"
        id="stat-pending-review"
        onClick={() => handleCardClick('pending_review')}
        className={`border-l-4 border-indigo-400 bg-indigo-50/40 px-3 py-2 rounded-r shadow-2xs text-left cursor-pointer hover:-translate-y-0.5 hover:shadow-sm transition-all ${
          activeStatFilter === 'pending_review' ? 'ring-2 ring-indigo-500 ring-offset-1 bg-indigo-100/40' : ''
        }`}
      >
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl sm:text-2xl font-black text-indigo-900">{pendingReview}</p>
          <span className="text-[9px] text-indigo-400 font-bold whitespace-nowrap">Submitted</span>
        </div>
        <p className="text-[10px] uppercase tracking-wider font-extrabold text-indigo-700 leading-tight mt-0.5">Pending Review</p>
      </button>

      {/* 5. Pending Approval */}
      <button
        type="button"
        id="stat-pending-approval"
        onClick={() => handleCardClick('pending_approval')}
        className={`border-l-4 border-cyan-500 bg-cyan-50/30 px-3 py-2 rounded-r shadow-2xs text-left cursor-pointer hover:-translate-y-0.5 hover:shadow-sm transition-all ${
          activeStatFilter === 'pending_approval' ? 'ring-2 ring-cyan-500 ring-offset-1 bg-cyan-100/30' : ''
        }`}
      >
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl sm:text-2xl font-black text-cyan-900">{pendingApproval}</p>
          <span className="text-[9px] text-cyan-500 font-bold whitespace-nowrap">Reviewed</span>
        </div>
        <p className="text-[10px] uppercase tracking-wider font-extrabold text-cyan-705 leading-tight mt-0.5">Pending Approval</p>
      </button>

      {/* 6. Approved */}
      <button
        type="button"
        id="stat-approved"
        onClick={() => handleCardClick('approved')}
        className={`border-l-4 border-emerald-500 bg-emerald-50/50 px-3 py-2 rounded-r shadow-2xs text-left cursor-pointer hover:-translate-y-0.5 hover:shadow-sm transition-all ${
          activeStatFilter === 'approved' ? 'ring-2 ring-emerald-500 ring-offset-1 bg-emerald-100/50' : ''
        }`}
      >
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl sm:text-2xl font-black text-emerald-800">{approved}</p>
          <span className="text-[9px] text-emerald-500 font-bold">
            ({totalObservations ? Math.round((approved / totalObservations) * 100) : 0}%)
          </span>
        </div>
        <p className="text-[10px] uppercase tracking-wider font-extrabold text-emerald-705 leading-tight mt-0.5">Approved</p>
      </button>
    </div>
  );
}
