'use client';

import { useState } from 'react';
import type { SignalNode, Resolution } from '@/lib/types';

interface DetailPanelProps {
  node: SignalNode;
  resolution: Resolution | undefined;
  computedProbability: number | null;
  parentNode: SignalNode | null;
  pathToRoot: SignalNode[];
  isOpen: boolean;
  marketGap: number | null;
  onResolve: (nodeId: string, resolution: Resolution | null) => void;
  onToggle: () => void;
  onNavigateToNode: (nodeId: string) => void;
}

export function DetailPanel({
  node,
  resolution,
  computedProbability,
  parentNode,
  pathToRoot,
  isOpen,
  marketGap,
  onResolve,
  onToggle,
  onNavigateToNode,
}: DetailPanelProps) {
  const [showFullText, setShowFullText] = useState(false);

  // Use computed probability if available, otherwise fall back to base_rate
  const displayProb = computedProbability ?? node.base_rate;
  const prob = displayProb != null ? (displayProb * 100).toFixed(0) : '-';
  const source = node.probability_source || 'unknown';
  const resDate = node.resolution_date
    ? new Date(node.resolution_date).toLocaleDateString()
    : '-';
  const isRoot = !node.parent_id;
  const isLeaf = !node.children?.length;

  const handleResolve = (res: Resolution) => {
    onResolve(node.id, resolution === res ? null : res);
  };

  // Calculate days until resolution
  const daysUntilResolution = node.resolution_date
    ? Math.ceil((new Date(node.resolution_date).getTime() - Date.now()) / (1000 * 60 * 60 * 24))
    : null;

  return (
    <>
      {/* Collapsed tab */}
      {!isOpen && (
        <button className="detail-panel-tab" onClick={onToggle}>
          <span className="tab-icon">◀</span>
          <span className="tab-label">Details</span>
        </button>
      )}

      {/* Main panel */}
      <div className={`detail-panel ${isOpen ? 'open' : 'closed'}`}>
        <div className="detail-header">
          <span>Selected Signal</span>
          <button className="panel-toggle" onClick={onToggle} title="Collapse panel">
            ▶
          </button>
        </div>

        {/* Main info card */}
        <div className="detail-card">
          <div className="signal-question">
            {truncate(node.text, 100)}
            {node.text.length > 100 && (
              <button
                className="show-more-btn"
                onClick={() => setShowFullText(true)}
              >
                Show full
              </button>
            )}
          </div>
          <div className="signal-meta">
            <span className={`source-badge ${source}`}>{source}</span>
            {node.relationship_type === 'necessity' && (
              <span className="necessity-badge">🔒 Required</span>
            )}
            <span>Resolves: {resDate}</span>
            {daysUntilResolution !== null && daysUntilResolution > 0 && (
              <span className="days-badge">
                {daysUntilResolution}d
              </span>
            )}
          </div>
          <div className="prob-value">{prob}%</div>
          <div className="prob-bar">
            <div
              className={`prob-fill ${source === 'llm' ? 'llm' : 'market'}`}
              style={{ width: `${prob}%` }}
            />
          </div>
        </div>

        {/* Market comparison (for nodes with market data) */}
        {node.market_price != null && (
          <>
            <div className="section-label">Market Comparison</div>
            <div className="detail-card market-card">
              <div className="market-row">
                <span className="market-label">Market Price:</span>
                <span className="market-value">{(node.market_price * 100).toFixed(0)}%</span>
              </div>
              {node.market_platform && (
                <div className="market-row">
                  <span className="market-label">Platform:</span>
                  <span className="market-platform">
                    {node.market_url ? (
                      <a href={node.market_url} target="_blank" rel="noopener noreferrer">
                        {node.market_platform} ↗
                      </a>
                    ) : (
                      node.market_platform
                    )}
                  </span>
                </div>
              )}
              {marketGap != null && !node.is_leaf && (
                <div className="market-row">
                  <span className="market-label">Gap:</span>
                  <span className={`market-gap ${Math.abs(marketGap) <= 5 ? 'ok' : Math.abs(marketGap) <= 15 ? 'warning' : 'review'}`}>
                    {marketGap >= 0 ? '+' : ''}{marketGap.toFixed(1)}pp
                    {Math.abs(marketGap) > 15 && ' ⚠️'}
                  </span>
                </div>
              )}
              {node.market_match_confidence != null && (
                <div className="market-row">
                  <span className="market-label">Match:</span>
                  <span className="market-confidence">
                    {(node.market_match_confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </>
        )}

        {/* Relationship to parent */}
        {!isRoot && node.rho != null && (
          <>
            <div className="section-label">Relationship to Parent</div>
            <div className="detail-card">
              <div className={`rho-display ${node.rho > 0 ? 'enhances' : 'suppresses'}`}>
                {node.rho > 0 ? 'Enhances' : 'Suppresses'} &nbsp;
                ρ = {node.rho > 0 ? '+' : ''}{node.rho.toFixed(2)}
              </div>
              {node.rho_reasoning && (
                <div className="rho-reasoning">
                  &ldquo;{truncateReasoning(node.rho_reasoning)}&rdquo;
                </div>
              )}
            </div>
          </>
        )}

        {/* Impact on parent (cruxiness) */}
        {!isRoot && node.p_parent_given_yes != null && node.p_parent_given_no != null && (
          <>
            <div className="section-label">Impact on Parent</div>
            <div className="detail-card">
              <div className="cruxiness-context">
                P(&ldquo;{truncate(parentNode?.text || 'parent', 30)}&rdquo;) if this resolves...
              </div>
              <CruxinessBar
                pYes={node.p_parent_given_yes}
                pNo={node.p_parent_given_no}
                baseline={parentNode?.base_rate ?? 0.5}
              />
            </div>
          </>
        )}

        {/* Path to root - clickable */}
        {!isRoot && pathToRoot.length > 1 && (
          <>
            <div className="section-label">Path to Root</div>
            <div className="detail-card path-card">
              {pathToRoot.map((pathNode, i) => (
                <div key={pathNode.id} className="path-step">
                  <button
                    className="path-node-btn"
                    onClick={() => onNavigateToNode(pathNode.id)}
                    title={`Click to select: ${pathNode.text}`}
                  >
                    {truncate(pathNode.text, 25)}
                  </button>
                  {i < pathToRoot.length - 1 && pathNode.rho != null && (
                    <span className={`path-rho ${pathNode.rho > 0 ? 'enhances' : 'suppresses'}`}>
                      ρ={pathNode.rho > 0 ? '+' : ''}{pathNode.rho.toFixed(2)}
                    </span>
                  )}
                  {i < pathToRoot.length - 1 && <span className="path-arrow">→</span>}
                </div>
              ))}
            </div>
          </>
        )}

        {/* What-if buttons for leaves */}
        {isLeaf && (
          <>
            <div className="section-label">What-If Analysis</div>
            <div className="detail-card whatif-card">
              <div className="whatif-buttons">
                <button
                  className={`whatif-btn yes ${resolution === 'yes' ? 'active' : ''}`}
                  onClick={() => handleResolve('yes')}
                >
                  {resolution === 'yes' ? '✓ ' : ''}YES
                </button>
                <button
                  className={`whatif-btn no ${resolution === 'no' ? 'active' : ''}`}
                  onClick={() => handleResolve('no')}
                >
                  {resolution === 'no' ? '✗ ' : ''}NO
                </button>
              </div>
              <div className="whatif-hint">
                {resolution
                  ? 'Click active button to reset'
                  : 'Resolve this signal to see impact on target'}
              </div>
            </div>
          </>
        )}

        {/* Resolution timeline */}
        {node.resolution_date && (
          <>
            <div className="section-label">Resolution Timeline</div>
            <div className="detail-card timeline-card">
              <ResolutionTimeline
                resolutionDate={node.resolution_date}
                daysUntil={daysUntilResolution}
              />
            </div>
          </>
        )}
      </div>

      {/* Full text modal */}
      {showFullText && (
        <div className="modal-overlay" onClick={() => setShowFullText(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Full Signal Question</h3>
              <button className="modal-close" onClick={() => setShowFullText(false)}>×</button>
            </div>
            <div className="modal-body">
              <p>{node.text}</p>
            </div>
            <div className="modal-footer">
              <div className="modal-meta">
                <span className={`source-badge ${source}`}>{source}</span>
                <span>P(YES): {prob}%</span>
                <span>Resolves: {resDate}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// Cruxiness visualization
function CruxinessBar({
  pYes,
  pNo,
  baseline,
}: {
  pYes: number;
  pNo: number;
  baseline: number;
}) {
  const yesPos = pYes * 100;
  const noPos = pNo * 100;
  const baselinePos = baseline * 100;
  const spreadLeft = Math.min(yesPos, noPos);
  const spreadWidth = Math.abs(yesPos - noPos);

  return (
    <>
      <div className="cruxiness-track">
        <div
          className="cruxiness-spread"
          style={{ left: `${spreadLeft}%`, width: `${spreadWidth}%` }}
        />
        <div
          className="cruxiness-marker baseline"
          style={{ left: `${baselinePos}%` }}
          title="Current P(parent)"
        />
        <div
          className="cruxiness-marker no"
          style={{ left: `${noPos}%` }}
        />
        <div
          className="cruxiness-marker yes"
          style={{ left: `${yesPos}%` }}
        />
      </div>
      <div className="cruxiness-labels">
        <span className="no">NO → {(pNo * 100).toFixed(0)}%</span>
        <span className="yes">YES → {(pYes * 100).toFixed(0)}%</span>
      </div>
      <div className="cruxiness-spread-label">
        Spread: {(spreadWidth).toFixed(0)}pp
      </div>
    </>
  );
}

// Resolution timeline visualization
function ResolutionTimeline({
  resolutionDate,
  daysUntil,
}: {
  resolutionDate: string;
  daysUntil: number | null;
}) {
  const resDate = new Date(resolutionDate);
  const now = new Date();
  const totalDays = 365; // Show 1 year timeline
  const progress = daysUntil !== null
    ? Math.max(0, Math.min(100, ((totalDays - daysUntil) / totalDays) * 100))
    : 100;

  return (
    <div className="timeline">
      <div className="timeline-track">
        <div className="timeline-progress" style={{ width: `${progress}%` }} />
        <div className="timeline-marker" style={{ left: `${progress}%` }} />
      </div>
      <div className="timeline-labels">
        <span>Now</span>
        <span className={daysUntil !== null && daysUntil < 30 ? 'soon' : ''}>
          {resDate.toLocaleDateString()}
          {daysUntil !== null && daysUntil > 0 && ` (${daysUntil}d)`}
          {daysUntil !== null && daysUntil <= 0 && ' (Past)'}
        </span>
      </div>
    </div>
  );
}

function truncate(text: string, maxLen: number): string {
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '…' : text;
}

function truncateReasoning(text: string): string {
  const directionMatch = text.match(/Direction:\s*([^|]+)/);
  if (directionMatch) {
    const direction = directionMatch[1].trim();
    return direction.length > 150 ? direction.slice(0, 150) + '…' : direction;
  }
  return text.length > 150 ? text.slice(0, 150) + '…' : text;
}
