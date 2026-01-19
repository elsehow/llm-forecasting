'use client';

import { useState } from 'react';

interface SensitivityItem {
  nodeId: string;
  text: string;
  currentProb: number;
  impactIfYes: number;
  impactIfNo: number;
  totalSwing: number;
}

interface SensitivityPanelProps {
  items: SensitivityItem[];
  onSelectNode: (nodeId: string) => void;
  onResolve: (nodeId: string, resolution: 'yes' | 'no') => void;
}

export function SensitivityPanel({
  items,
  onSelectNode,
  onResolve,
}: SensitivityPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (items.length === 0) {
    return null;
  }

  const displayItems = isExpanded ? items : items.slice(0, 5);
  const maxSwing = items[0]?.totalSwing ?? 0;

  return (
    <div className="sensitivity-panel">
      <div className="sensitivity-header">
        <span className="sensitivity-title">Sensitivity Analysis</span>
        <span className="sensitivity-subtitle">
          Which unresolved signals have the highest impact?
        </span>
      </div>

      <div className="sensitivity-list">
        {displayItems.map((item, index) => {
          const swingWidth = maxSwing > 0 ? (item.totalSwing / maxSwing) * 100 : 0;
          const currentProbPct = (item.currentProb * 100).toFixed(0);
          const swingPct = (item.totalSwing * 100).toFixed(0);
          const yesImpact = item.impactIfYes >= 0 ? `+${(item.impactIfYes * 100).toFixed(0)}` : (item.impactIfYes * 100).toFixed(0);
          const noImpact = item.impactIfNo >= 0 ? `+${(item.impactIfNo * 100).toFixed(0)}` : (item.impactIfNo * 100).toFixed(0);

          return (
            <div
              key={item.nodeId}
              className="sensitivity-row"
              onClick={() => onSelectNode(item.nodeId)}
            >
              <div className="sensitivity-rank">#{index + 1}</div>
              <div className="sensitivity-content">
                <div className="sensitivity-text" title={item.text}>
                  {truncate(item.text, 50)}
                </div>
                <div className="sensitivity-meta">
                  <span className="current-prob">P: {currentProbPct}%</span>
                  <span className="swing-value">Swing: {swingPct}pp</span>
                </div>
                <div className="sensitivity-bar-track">
                  <div
                    className="sensitivity-bar-fill"
                    style={{ width: `${swingWidth}%` }}
                  />
                </div>
              </div>
              <div className="sensitivity-impacts">
                <button
                  className="impact-btn yes"
                  onClick={(e) => { e.stopPropagation(); onResolve(item.nodeId, 'yes'); }}
                  title={`If YES: ${yesImpact}pp to target`}
                >
                  {yesImpact}pp
                </button>
                <button
                  className="impact-btn no"
                  onClick={(e) => { e.stopPropagation(); onResolve(item.nodeId, 'no'); }}
                  title={`If NO: ${noImpact}pp to target`}
                >
                  {noImpact}pp
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {items.length > 5 && (
        <button
          className="expand-btn"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? 'Show less' : `Show all ${items.length}`}
        </button>
      )}
    </div>
  );
}

function truncate(text: string, maxLen: number): string {
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '…' : text;
}
