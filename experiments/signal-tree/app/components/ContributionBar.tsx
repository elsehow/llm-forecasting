'use client';

import { useState } from 'react';
import type { ContributionData } from '@/lib/types';

interface ContributionBarProps {
  contributions: ContributionData[];
  selectedNodeId: string | null;
  showEnhancersOnly: boolean;
  showSuppressorsOnly: boolean;
  onSelectNode: (nodeId: string) => void;
  onHighlightPath: (nodeId: string) => void;
  onClearHighlight: () => void;
  onFilterChange: (enhancersOnly: boolean, suppressorsOnly: boolean) => void;
}

export function ContributionBar({
  contributions,
  selectedNodeId,
  showEnhancersOnly,
  showSuppressorsOnly,
  onSelectNode,
  onHighlightPath,
  onClearHighlight,
  onFilterChange,
}: ContributionBarProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Filter contributions based on direction
  const filteredContributions = contributions.filter(contrib => {
    if (showEnhancersOnly && contrib.direction !== 'enhances') return false;
    if (showSuppressorsOnly && contrib.direction !== 'suppresses') return false;
    return true;
  });

  // Show top 6 or all if expanded
  const displayContributions = isExpanded
    ? filteredContributions
    : filteredContributions.slice(0, 6);

  const maxContrib = filteredContributions[0]?.contribution ?? 0;

  return (
    <div className="contribution-bar">
      <div className="contribution-header">
        <span className="contribution-title">
          Top Contributors (Leaves)
          {filteredContributions.length > 0 && (
            <span className="contribution-count">
              {filteredContributions.length} signals
            </span>
          )}
        </span>
        <div className="contribution-filters">
          <button
            className={`filter-btn ${showEnhancersOnly ? 'active enhances' : ''}`}
            onClick={() => onFilterChange(!showEnhancersOnly, false)}
            title="Show only enhancers"
          >
            ↑
          </button>
          <button
            className={`filter-btn ${showSuppressorsOnly ? 'active suppresses' : ''}`}
            onClick={() => onFilterChange(false, !showSuppressorsOnly)}
            title="Show only suppressors"
          >
            ↓
          </button>
          {(showEnhancersOnly || showSuppressorsOnly) && (
            <button
              className="filter-btn clear"
              onClick={() => onFilterChange(false, false)}
              title="Clear filters"
            >
              ×
            </button>
          )}
        </div>
      </div>

      <div className="contribution-list">
        {displayContributions.map(contrib => {
          const direction = contrib.direction;
          const barWidth = maxContrib > 0 ? (contrib.contribution / maxContrib) * 100 : 0;
          const contribPercent = (contrib.contribution * 100).toFixed(1);
          const isSelected = contrib.signal_id === selectedNodeId;
          const probPercent = (contrib.base_rate * 100).toFixed(0);

          return (
            <div
              key={contrib.signal_id}
              className={`contribution-row ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectNode(contrib.signal_id)}
              onMouseEnter={() => onHighlightPath(contrib.signal_id)}
              onMouseLeave={onClearHighlight}
            >
              <span className="contribution-name" title={contrib.text}>
                {truncate(contrib.text, 40)}
              </span>
              <span className="contribution-prob">{probPercent}%</span>
              <div className="contribution-bar-track">
                <div
                  className={`contribution-bar-fill ${direction}`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <span className={`contribution-value ${direction}`}>
                {direction === 'enhances' ? '+' : '−'}{contribPercent}%
              </span>
              <span className={`contribution-direction ${direction}`}>
                {direction === 'enhances' ? '↑' : '↓'}
              </span>
            </div>
          );
        })}
      </div>

      {filteredContributions.length > 6 && (
        <button
          className="expand-btn"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? 'Show less' : `Show all ${filteredContributions.length}`}
        </button>
      )}
    </div>
  );
}

function truncate(text: string, maxLen: number): string {
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '…' : text;
}
