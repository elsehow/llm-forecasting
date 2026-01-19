'use client';

import { useEffect, useState, useCallback } from 'react';
import { useTreeState } from '@/lib/useTreeState';
import { TreeCanvas } from '@/components/TreeCanvas';
import { DetailPanel } from '@/components/DetailPanel';
import { ContributionBar } from '@/components/ContributionBar';
import { Toolbar } from '@/components/Toolbar';
import { SensitivityPanel } from '@/components/SensitivityPanel';
import './styles.css';

interface TargetFile {
  id: string;
  label: string;
  path: string;
  timestamp: string;
  mtime: number;
  targetQuestion: string;
}

export default function Home() {
  const [targets, setTargets] = useState<TargetFile[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [showSensitivity, setShowSensitivity] = useState(false);

  const {
    data,
    selectedNodeId,
    resolutions,
    computedProbabilities,
    selectedNode,
    stats,
    collapsedNodes,
    scenarios,
    activeScenarioId,
    viewSettings,
    filterSettings,
    highlightedPath,
    batchSelectMode,
    batchSelectedNodes,
    filteredNodes,
    sensitivityAnalysis,
    dynamicContributions,
    loadData,
    selectNode,
    resolveSignal,
    resetAllResolutions,
    toggleCollapse,
    expandAll,
    collapseAll,
    setZoom,
    setPan,
    toggleDarkMode,
    toggleMinimap,
    toggleDetailPanel,
    setSearchQuery,
    setProbabilityFilter,
    setDirectionFilter,
    clearFilters,
    saveScenario,
    loadScenario,
    deleteScenario,
    toggleBatchMode,
    toggleBatchSelect,
    batchResolve,
    setHighlightedPath,
    clearHighlightedPath,
    findNode,
    getPathToRoot,
  } = useTreeState();

  // Discover available targets
  useEffect(() => {
    async function discoverTargets() {
      try {
        const response = await fetch('/api/targets');
        const data: TargetFile[] = await response.json();
        setTargets(data);

        if (data.length > 0 && !selectedTarget) {
          setSelectedTarget(data[0].path);
          loadData(data[0].path);
        }
      } catch (error) {
        console.error('Failed to discover targets:', error);
      }
    }
    discoverTargets();
  }, []);

  const handleTargetChange = useCallback((path: string) => {
    setSelectedTarget(path);
    loadData(path);
    resetAllResolutions();
  }, [loadData, resetAllResolutions]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape clears selection
      if (e.key === 'Escape') {
        selectNode(null);
        clearHighlightedPath();
      }
      // Ctrl/Cmd + F focuses search
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-input') as HTMLInputElement;
        searchInput?.focus();
      }
      // D toggles dark mode
      if (e.key === 'd' && !e.ctrlKey && !e.metaKey && !isInputFocused()) {
        toggleDarkMode();
      }
      // M toggles minimap
      if (e.key === 'm' && !e.ctrlKey && !e.metaKey && !isInputFocused()) {
        toggleMinimap();
      }
      // B toggles batch mode
      if (e.key === 'b' && !e.ctrlKey && !e.metaKey && !isInputFocused()) {
        toggleBatchMode();
      }
      // S toggles sensitivity view
      if (e.key === 's' && !e.ctrlKey && !e.metaKey && !isInputFocused()) {
        setShowSensitivity(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectNode, clearHighlightedPath, toggleDarkMode, toggleMinimap, toggleBatchMode]);

  // Get parent node for detail panel
  const parentNode = selectedNode?.parent_id
    ? findNode(selectedNode.parent_id)
    : null;

  // Get path to root for detail panel
  const pathToRoot = selectedNodeId ? getPathToRoot(selectedNodeId) : [];

  // Handle path highlight from contribution bar
  const handleHighlightPath = useCallback((nodeId: string) => {
    const path = getPathToRoot(nodeId);
    setHighlightedPath(path.map(n => n.id));
  }, [getPathToRoot, setHighlightedPath]);

  const hasResolutions = Object.keys(resolutions).length > 0;
  const currentTarget = targets.find(t => t.path === selectedTarget);

  return (
    <div className={`app ${viewSettings.darkMode ? 'dark' : ''}`}>
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1>Signal Tree Explorer</h1>
          <div className="selector-group">
            <span className="selector-label">Target</span>
            <select
              className="file-selector"
              value={selectedTarget ?? ''}
              onChange={(e) => e.target.value && handleTargetChange(e.target.value)}
              title={currentTarget?.targetQuestion}
            >
              {targets.map(t => (
                <option key={t.id} value={t.path}>{t.label}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="stats-bar">
          <div className="stat">
            Signals: <span className="stat-value">{stats?.totalSignals ?? '-'}</span>
          </div>
          <div className="stat">
            Leaves: <span className="stat-value">{stats?.leafCount ?? '-'}</span>
          </div>
          <div className="stat">
            Depth: <span className="stat-value">{stats?.maxDepth ?? '-'}</span>
          </div>
          <div className="stat">
            P(Target):{' '}
            <span className="stat-value">
              {stats?.currentProbability != null
                ? `${(stats.currentProbability * 100).toFixed(0)}%`
                : '-'}
              {stats?.hasResolutions && stats.originalProbability != null && (
                <span
                  className={`stat-delta ${
                    (stats.currentProbability ?? 0) >= stats.originalProbability
                      ? 'up'
                      : 'down'
                  }`}
                >
                  {' '}({((stats.currentProbability ?? 0) - stats.originalProbability) >= 0 ? '+' : ''}
                  {(((stats.currentProbability ?? 0) - stats.originalProbability) * 100).toFixed(0)}pp)
                </span>
              )}
            </span>
          </div>
          {hasResolutions && (
            <button className="reset-all-btn" onClick={resetAllResolutions}>
              Reset All ({stats?.resolutionCount})
            </button>
          )}
        </div>
      </header>

      {/* Toolbar */}
      <Toolbar
        searchQuery={filterSettings.searchQuery}
        probabilityMin={filterSettings.probabilityMin}
        probabilityMax={filterSettings.probabilityMax}
        onSearchChange={setSearchQuery}
        onProbabilityFilterChange={setProbabilityFilter}
        onClearFilters={clearFilters}
        batchSelectMode={batchSelectMode}
        batchSelectedCount={batchSelectedNodes.size}
        onToggleBatchMode={toggleBatchMode}
        onBatchResolve={batchResolve}
        scenarios={scenarios}
        activeScenarioId={activeScenarioId}
        onSaveScenario={saveScenario}
        onLoadScenario={loadScenario}
        onDeleteScenario={deleteScenario}
        darkMode={viewSettings.darkMode}
        showMinimap={viewSettings.showMinimap}
        onToggleDarkMode={toggleDarkMode}
        onToggleMinimap={toggleMinimap}
        onExpandAll={expandAll}
        onCollapseAll={collapseAll}
      />

      {/* Main content */}
      <div className="main-container">
        <div className="content-area">
          {/* Tree canvas */}
          {data?.tree.target && (
            <TreeCanvas
              root={data.tree.target}
              selectedNodeId={selectedNodeId}
              resolutions={resolutions}
              computedProbabilities={computedProbabilities}
              collapsedNodes={collapsedNodes}
              filteredNodes={filteredNodes}
              highlightedPath={highlightedPath}
              batchSelectMode={batchSelectMode}
              batchSelectedNodes={batchSelectedNodes}
              zoom={viewSettings.zoom}
              panX={viewSettings.panX}
              panY={viewSettings.panY}
              showMinimap={viewSettings.showMinimap}
              onNodeClick={selectNode}
              onNodeDoubleClick={toggleCollapse}
              onBatchSelect={toggleBatchSelect}
              onZoomChange={setZoom}
              onPanChange={setPan}
            />
          )}

          {/* Detail panel */}
          {selectedNode && (
            <DetailPanel
              node={selectedNode}
              resolution={resolutions[selectedNode.id]}
              parentNode={parentNode}
              pathToRoot={pathToRoot}
              isOpen={viewSettings.detailPanelOpen}
              onResolve={resolveSignal}
              onToggle={toggleDetailPanel}
              onNavigateToNode={selectNode}
            />
          )}

          {/* Collapsed detail panel tab when no selection */}
          {!selectedNode && !viewSettings.detailPanelOpen && (
            <button className="detail-panel-tab" onClick={toggleDetailPanel}>
              <span className="tab-icon">◀</span>
              <span className="tab-label">Details</span>
            </button>
          )}
        </div>

        {/* Bottom panels */}
        <div className="bottom-panels">
          {/* Contribution bar */}
          {dynamicContributions.length > 0 && (
            <ContributionBar
              contributions={dynamicContributions}
              selectedNodeId={selectedNodeId}
              showEnhancersOnly={filterSettings.showEnhancersOnly}
              showSuppressorsOnly={filterSettings.showSuppressorsOnly}
              onSelectNode={selectNode}
              onHighlightPath={handleHighlightPath}
              onClearHighlight={clearHighlightedPath}
              onFilterChange={setDirectionFilter}
            />
          )}

          {/* Sensitivity panel toggle */}
          <button
            className={`sensitivity-toggle ${showSensitivity ? 'active' : ''}`}
            onClick={() => setShowSensitivity(!showSensitivity)}
          >
            {showSensitivity ? '▼ Sensitivity' : '▶ Sensitivity (S)'}
          </button>

          {/* Sensitivity panel */}
          {showSensitivity && sensitivityAnalysis.length > 0 && (
            <SensitivityPanel
              items={sensitivityAnalysis}
              onSelectNode={selectNode}
              onResolve={resolveSignal}
            />
          )}
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="keyboard-hints">
        <span>D: Dark mode</span>
        <span>M: Minimap</span>
        <span>B: Batch mode</span>
        <span>S: Sensitivity</span>
        <span>Esc: Clear selection</span>
        <span>Ctrl+F: Search</span>
      </div>
    </div>
  );
}

function isInputFocused(): boolean {
  const active = document.activeElement;
  return active?.tagName === 'INPUT' || active?.tagName === 'TEXTAREA' || active?.tagName === 'SELECT';
}
