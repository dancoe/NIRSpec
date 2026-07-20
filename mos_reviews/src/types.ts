/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ProgramReview {
  id: string;
  cycle?: string;
  pi: string;
  aptPrep: string;
  program: string;
  observation: string;
  programInfoUrl: string;
  visitStatusUrl: string;
  offset: string;
  submissionDueDate: string;
  submittedDate: string;
  reviewDueDate?: string;
  reviewedDate: string;
  finalizeDueDate: string;
  approvedDate: string;
  sparDeadline: string;
  flightReadyEarliest: string;
  flightReadyMid: string;
  flightReadyLatest: string;
  obsEarliest: string;
  obsMid: string;
  obsLatest: string;
  isCompleted: boolean;
  notes?: string;
  isDeleted?: boolean;
  nirspecReviewer?: string;
  nircamReviewer?: string;
  miriReviewer?: string;
  nirissReviewer?: string;
}

export type CompletionFilter = 'all' | 'pending' | 'completed' | 'deleted';

export interface ColumnConfig {
  key: string;
  label: string;
  width: number;
  visible: boolean;
}

export const DEFAULT_COLUMNS: ColumnConfig[] = [
  { key: 'select', label: '🔘', width: 45, visible: true },
  { key: 'cycle', label: 'Cycle', width: 55, visible: true },
  { key: 'program', label: 'Program', width: 85, visible: true },
  { key: 'observation', label: 'Obs #', width: 65, visible: true },
  { key: 'pi', label: 'PI', width: 130, visible: true },
  { key: 'aptPrep', label: 'APT Prep', width: 95, visible: true },
  { key: 'nirspecReviewer', label: 'NIRSpec Rev', width: 105, visible: true },
  { key: 'nircamReviewer', label: 'NIRCam Rev', width: 105, visible: false },
  { key: 'miriReviewer', label: 'MIRI Rev', width: 105, visible: false },
  { key: 'nirissReviewer', label: 'NIRISS Rev', width: 105, visible: false },
  { key: 'notes', label: 'Notes', width: 160, visible: true },
  { key: 'submissionDue', label: '🗓️SUBMIT', width: 100, visible: true },
  { key: 'submittedDate', label: '☑️SUBMITTED', width: 100, visible: true },
  { key: 'reviewDue', label: '🗓️REVIEW', width: 100, visible: true },
  { key: 'reviewedDate', label: '☑️REVIEWED', width: 100, visible: true },
  { key: 'finalizeDue', label: '🗓️FINALIZE', width: 100, visible: true },
  { key: 'closed', label: '☑️APPROVED', width: 100, visible: true },
  { key: 'flightReady', label: '🗓️FLIGHT READY', width: 120, visible: true },
  { key: 'planWindow', label: '🗓️PLAN WINDOW', width: 220, visible: true },
  { key: 'actions', label: 'Actions', width: 95, visible: true },
];

export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}
