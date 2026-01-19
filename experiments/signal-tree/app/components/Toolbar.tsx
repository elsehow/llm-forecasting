'use client';

import type { Scenario, Resolution } from '@/lib/types';

interface ToolbarProps {
  // Search & Filter
  searchQuery: string;
  probabilityMin: number;
  probabilityMax: number;
  onSearchChange: (query: string) => void;
  onProbabilityFilterChange: (min: number, max: number) => void;
  onClearFilters: () => void;

  // Batch mode
  batchSelectMode: boolean;
  batchSelectedCount: number;
  onToggleBatchMode: () => void;
  onBatchResolve: (resolution: Resolution) => void;

  // Scenarios
  scenarios: Scenario[];
  activeScenarioId: string | null;
  onSaveScenario: (name: string) => void;
  onLoadScenario: (id: string) => void;
  onDeleteScenario: (id: string) => void;

  // View
  darkMode: boolean;
  showMinimap: boolean;
  onToggleDarkMode: () => void;
  onToggleMinimap: () => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
}

export function Toolbar({
  searchQuery,
  probabilityMin,
  probabilityMax,
  onSearchChange,
  onProbabilityFilterChange,
  onClearFilters,
  batchSelectMode,
  batchSelectedCount,
  onToggleBatchMode,
  onBatchResolve,
  scenarios,
  activeScenarioId,
  onSaveScenario,
  onLoadScenario,
  onDeleteScenario,
  darkMode,
  showMinimap,
  onToggleDarkMode,
  onToggleMinimap,
  onExpandAll,
  onCollapseAll,
}: ToolbarProps) {
  const hasFilters = searchQuery || probabilityMin > 0 || probabilityMax < 100;

  const handleSaveScenario = () => {
    const name = prompt('Enter scenario name:', `Scenario ${scenarios.length + 1}`);
    if (name) {
      onSaveScenario(name);
    }
  };

  return (
    <div className="toolbar">
      {/* Search */}
      <div className="toolbar-section search-section">
        <input
          type="text"
          className="search-input"
          placeholder="Search signals..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
        {searchQuery && (
          <button
            className="clear-search"
            onClick={() => onSearchChange('')}
          >
            ×
          </button>
        )}
      </div>

      {/* Probability Filter */}
      <div className="toolbar-section prob-filter-section">
        <span className="filter-label">P:</span>
        <input
          type="number"
          className="prob-input"
          min={0}
          max={100}
          value={probabilityMin}
          onChange={(e) => onProbabilityFilterChange(Number(e.target.value), probabilityMax)}
        />
        <span className="filter-dash">-</span>
        <input
          type="number"
          className="prob-input"
          min={0}
          max={100}
          value={probabilityMax}
          onChange={(e) => onProbabilityFilterChange(probabilityMin, Number(e.target.value))}
        />
        <span className="filter-label">%</span>
        {hasFilters && (
          <button className="clear-filters-btn" onClick={onClearFilters} title="Clear all filters">
            Clear
          </button>
        )}
      </div>

      {/* Batch Mode */}
      <div className="toolbar-section batch-section">
        <button
          className={`batch-mode-btn ${batchSelectMode ? 'active' : ''}`}
          onClick={onToggleBatchMode}
          title="Toggle batch selection mode"
        >
          {batchSelectMode ? '✓ Batch' : 'Batch'}
        </button>
        {batchSelectMode && batchSelectedCount > 0 && (
          <>
            <span className="batch-count">{batchSelectedCount} selected</span>
            <button
              className="batch-resolve-btn yes"
              onClick={() => onBatchResolve('yes')}
            >
              All YES
            </button>
            <button
              className="batch-resolve-btn no"
              onClick={() => onBatchResolve('no')}
            >
              All NO
            </button>
          </>
        )}
      </div>

      {/* Scenarios */}
      <div className="toolbar-section scenario-section">
        <button
          className="save-scenario-btn"
          onClick={handleSaveScenario}
          title="Save current resolutions as scenario"
        >
          Save
        </button>
        {scenarios.length > 0 && (
          <select
            className="scenario-select"
            value={activeScenarioId || ''}
            onChange={(e) => e.target.value && onLoadScenario(e.target.value)}
          >
            <option value="">Load scenario...</option>
            {scenarios.map(s => (
              <option key={s.id} value={s.id}>
                {s.name} ({Object.keys(s.resolutions).length} res.)
              </option>
            ))}
          </select>
        )}
        {activeScenarioId && (
          <button
            className="delete-scenario-btn"
            onClick={() => onDeleteScenario(activeScenarioId)}
            title="Delete current scenario"
          >
            ×
          </button>
        )}
      </div>

      {/* View Controls */}
      <div className="toolbar-section view-section">
        <button
          className="view-btn"
          onClick={onExpandAll}
          title="Expand all branches"
        >
          ⊞
        </button>
        <button
          className="view-btn"
          onClick={onCollapseAll}
          title="Collapse all branches"
        >
          ⊟
        </button>
        <button
          className={`view-btn ${showMinimap ? 'active' : ''}`}
          onClick={onToggleMinimap}
          title="Toggle minimap"
        >
          ⊡
        </button>
        <button
          className={`view-btn dark-mode-btn ${darkMode ? 'active' : ''}`}
          onClick={onToggleDarkMode}
          title="Toggle dark mode"
        >
          {darkMode ? '☀' : '☾'}
        </button>
      </div>
    </div>
  );
}
