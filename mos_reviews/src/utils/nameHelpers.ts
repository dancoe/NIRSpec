/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export function formatPiName(name: string): { last: string; first: string } {
  if (!name) return { last: '', first: '' };
  const trimmed = name.trim();
  if (trimmed.includes(',')) {
    const parts = trimmed.split(',');
    return {
      last: parts[0].trim(),
      first: parts.slice(1).join(',').trim()
    };
  }
  const lastSpace = trimmed.lastIndexOf(' ');
  if (lastSpace === -1) {
    return { last: trimmed, first: '' };
  }
  return {
    last: trimmed.substring(lastSpace + 1),
    first: trimmed.substring(0, lastSpace)
  };
}
