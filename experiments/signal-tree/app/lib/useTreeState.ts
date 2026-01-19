'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import type {
  TreeDataFile,
  SignalNode,
  Resolution,
  Scenario,
  ViewSettings,
  FilterSettings,
  ContributionData,
} from './types';

interface TreeState {
  data: TreeDataFile | null;
  selectedNodeId: string | null;
  resolutions: Record<string, Resolution>;
  collapsedNodes: Set<string>;
  scenarios: Scenario[];
  activeScenarioId: string | null;
  viewSettings: ViewSettings;
  filterSettings: FilterSettings;
  highlightedPath: string[];
  batchSelectMode: boolean;
  batchSelectedNodes: Set<string>;
}

const DEFAULT_VIEW_SETTINGS: ViewSettings = {
  zoom: 1,
  panX: 0,
  panY: 0,
  darkMode: false,
  showMinimap: true,
  detailPanelOpen: true,
};

const DEFAULT_FILTER_SETTINGS: FilterSettings = {
  searchQuery: '',
  probabilityMin: 0,
  probabilityMax: 100,
  showEnhancersOnly: false,
  showSuppressorsOnly: false,
};

export function useTreeState() {
  const [state, setState] = useState<TreeState>({
    data: null,
    selectedNodeId: null,
    resolutions: {},
    collapsedNodes: new Set(),
    scenarios: [],
    activeScenarioId: null,
    viewSettings: DEFAULT_VIEW_SETTINGS,
    filterSettings: DEFAULT_FILTER_SETTINGS,
    highlightedPath: [],
    batchSelectMode: false,
    batchSelectedNodes: new Set(),
  });

  // Load dark mode preference from localStorage
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode !== null) {
      setState(prev => ({
        ...prev,
        viewSettings: { ...prev.viewSettings, darkMode: savedDarkMode === 'true' },
      }));
    }
  }, []);

  // Apply dark mode to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', state.viewSettings.darkMode);
    localStorage.setItem('darkMode', String(state.viewSettings.darkMode));
  }, [state.viewSettings.darkMode]);

  // Load tree data from JSON - handles multiple formats
  const loadData = useCallback(async (url: string) => {
    const response = await fetch(url);
    const rawData = await response.json();

    // Normalize to TreeDataFile format
    let data: TreeDataFile;

    if (rawData.tree && rawData.tree.target) {
      // Old format: { config, tree: { target }, analysis }
      data = rawData as TreeDataFile;
    } else if (rawData.target) {
      // New format: { target } - synthesize the rest
      const target = rawData.target as SignalNode;

      // Count signals and find max depth
      const collectSignals = (node: SignalNode): SignalNode[] => {
        const signals: SignalNode[] = [];
        for (const child of node.children) {
          signals.push(child);
          signals.push(...collectSignals(child));
        }
        return signals;
      };

      const signals = collectSignals(target);
      const maxDepth = signals.reduce((max, s) => Math.max(max, s.depth || 0), 0);
      const leafCount = signals.filter(s => s.is_leaf).length;

      data = {
        config: {
          target_question: target.text,
          target_id: target.id,
          minimum_resolution_days: 7,
          max_signals: signals.length,
          actionable_horizon_days: 60,
          signals_per_node: 5,
          generation_model: 'unknown',
          rho_model: 'unknown',
          include_market_signals: false,
          market_match_threshold: 0.6,
        },
        tree: {
          target,
          signals,
          max_depth: maxDepth,
          leaf_count: leafCount,
          actionable_horizon_days: 60,
          computed_probability: null,
        },
        analysis: {
          target: target.text,
          computed_probability: 0.5,
          target_prior: 0.5,
          max_depth: maxDepth,
          total_signals: signals.length,
          leaf_count: leafCount,
          top_contributors: [],
          all_contributions: [],
        },
        generated_at: new Date().toISOString(),
      };
    } else {
      throw new Error('Unknown JSON format: expected tree.target or target');
    }

    setState(prev => ({
      ...prev,
      data,
      selectedNodeId: null,
      resolutions: {},
      collapsedNodes: new Set(),
      highlightedPath: [],
      batchSelectedNodes: new Set(),
    }));
  }, []);

  // Select/deselect a node
  const selectNode = useCallback((nodeId: string | null) => {
    setState(prev => ({
      ...prev,
      selectedNodeId: prev.selectedNodeId === nodeId ? null : nodeId,
    }));
  }, []);

  // Resolve a signal YES/NO or clear it
  const resolveSignal = useCallback((nodeId: string, resolution: Resolution | null) => {
    setState(prev => {
      const newResolutions = { ...prev.resolutions };
      if (resolution === null) {
        delete newResolutions[nodeId];
      } else {
        newResolutions[nodeId] = resolution;
      }
      return { ...prev, resolutions: newResolutions };
    });
  }, []);

  // Reset all resolutions
  const resetAllResolutions = useCallback(() => {
    setState(prev => ({ ...prev, resolutions: {}, activeScenarioId: null }));
  }, []);

  // Toggle node collapse
  const toggleCollapse = useCallback((nodeId: string) => {
    setState(prev => {
      const newCollapsed = new Set(prev.collapsedNodes);
      if (newCollapsed.has(nodeId)) {
        newCollapsed.delete(nodeId);
      } else {
        newCollapsed.add(nodeId);
      }
      return { ...prev, collapsedNodes: newCollapsed };
    });
  }, []);

  // Expand all nodes
  const expandAll = useCallback(() => {
    setState(prev => ({ ...prev, collapsedNodes: new Set() }));
  }, []);

  // Collapse all non-root nodes
  const collapseAll = useCallback(() => {
    if (!state.data) return;
    const toCollapse = new Set<string>();
    const collectInternal = (node: SignalNode) => {
      if (node.children?.length) {
        toCollapse.add(node.id);
        node.children.forEach(collectInternal);
      }
    };
    // Don't collapse root
    state.data.tree.target.children?.forEach(collectInternal);
    setState(prev => ({ ...prev, collapsedNodes: toCollapse }));
  }, [state.data]);

  // View settings
  const setZoom = useCallback((zoom: number) => {
    setState(prev => ({
      ...prev,
      viewSettings: { ...prev.viewSettings, zoom: Math.max(0.25, Math.min(2, zoom)) },
    }));
  }, []);

  const setPan = useCallback((panX: number, panY: number) => {
    setState(prev => ({
      ...prev,
      viewSettings: { ...prev.viewSettings, panX, panY },
    }));
  }, []);

  const resetView = useCallback(() => {
    setState(prev => ({
      ...prev,
      viewSettings: { ...prev.viewSettings, zoom: 1, panX: 0, panY: 0 },
    }));
  }, []);

  const toggleDarkMode = useCallback(() => {
    setState(prev => ({
      ...prev,
      viewSettings: { ...prev.viewSettings, darkMode: !prev.viewSettings.darkMode },
    }));
  }, []);

  const toggleMinimap = useCallback(() => {
    setState(prev => ({
      ...prev,
      viewSettings: { ...prev.viewSettings, showMinimap: !prev.viewSettings.showMinimap },
    }));
  }, []);

  const toggleDetailPanel = useCallback(() => {
    setState(prev => ({
      ...prev,
      viewSettings: { ...prev.viewSettings, detailPanelOpen: !prev.viewSettings.detailPanelOpen },
    }));
  }, []);

  // Filter settings
  const setSearchQuery = useCallback((query: string) => {
    setState(prev => ({
      ...prev,
      filterSettings: { ...prev.filterSettings, searchQuery: query },
    }));
  }, []);

  const setProbabilityFilter = useCallback((min: number, max: number) => {
    setState(prev => ({
      ...prev,
      filterSettings: { ...prev.filterSettings, probabilityMin: min, probabilityMax: max },
    }));
  }, []);

  const setDirectionFilter = useCallback((enhancersOnly: boolean, suppressorsOnly: boolean) => {
    setState(prev => ({
      ...prev,
      filterSettings: {
        ...prev.filterSettings,
        showEnhancersOnly: enhancersOnly,
        showSuppressorsOnly: suppressorsOnly,
      },
    }));
  }, []);

  const clearFilters = useCallback(() => {
    setState(prev => ({ ...prev, filterSettings: DEFAULT_FILTER_SETTINGS }));
  }, []);

  // Scenario management
  const saveScenario = useCallback((name: string) => {
    const newScenario: Scenario = {
      id: `scenario_${Date.now()}`,
      name,
      resolutions: { ...state.resolutions },
      createdAt: Date.now(),
    };
    setState(prev => ({
      ...prev,
      scenarios: [...prev.scenarios, newScenario],
      activeScenarioId: newScenario.id,
    }));
    return newScenario.id;
  }, [state.resolutions]);

  const loadScenario = useCallback((scenarioId: string) => {
    const scenario = state.scenarios.find(s => s.id === scenarioId);
    if (scenario) {
      setState(prev => ({
        ...prev,
        resolutions: { ...scenario.resolutions },
        activeScenarioId: scenarioId,
      }));
    }
  }, [state.scenarios]);

  const deleteScenario = useCallback((scenarioId: string) => {
    setState(prev => ({
      ...prev,
      scenarios: prev.scenarios.filter(s => s.id !== scenarioId),
      activeScenarioId: prev.activeScenarioId === scenarioId ? null : prev.activeScenarioId,
    }));
  }, []);

  // Batch selection
  const toggleBatchMode = useCallback(() => {
    setState(prev => ({
      ...prev,
      batchSelectMode: !prev.batchSelectMode,
      batchSelectedNodes: new Set(),
    }));
  }, []);

  const toggleBatchSelect = useCallback((nodeId: string) => {
    setState(prev => {
      const newSelected = new Set(prev.batchSelectedNodes);
      if (newSelected.has(nodeId)) {
        newSelected.delete(nodeId);
      } else {
        newSelected.add(nodeId);
      }
      return { ...prev, batchSelectedNodes: newSelected };
    });
  }, []);

  const batchResolve = useCallback((resolution: Resolution) => {
    setState(prev => {
      const newResolutions = { ...prev.resolutions };
      prev.batchSelectedNodes.forEach(nodeId => {
        newResolutions[nodeId] = resolution;
      });
      return {
        ...prev,
        resolutions: newResolutions,
        batchSelectedNodes: new Set(),
        batchSelectMode: false,
      };
    });
  }, []);

  // Highlight path (for contribution bar clicks)
  const setHighlightedPath = useCallback((nodeIds: string[]) => {
    setState(prev => ({ ...prev, highlightedPath: nodeIds }));
  }, []);

  const clearHighlightedPath = useCallback(() => {
    setState(prev => ({ ...prev, highlightedPath: [] }));
  }, []);

  // Compute probabilities based on resolutions (derived state)
  const computedProbabilities = useMemo(() => {
    if (!state.data) return {};

    const probs: Record<string, number> = {};
    const targetPrior = state.data.analysis?.target_prior ?? 0.5;

    const computeNodeProb = (node: SignalNode, prior: number = 0.5): number => {
      const resolution = state.resolutions[node.id];
      if (resolution === 'yes') {
        probs[node.id] = 1.0;
        return 1.0;
      }
      if (resolution === 'no') {
        probs[node.id] = 0.0;
        return 0.0;
      }

      if (!node.children || node.children.length === 0) {
        const prob = node.base_rate ?? prior;
        probs[node.id] = prob;
        return prob;
      }

      const safePrior = Math.max(0.01, Math.min(0.99, prior));
      let logOdds = Math.log(safePrior / (1 - safePrior));
      const k = 2.0;

      for (const child of node.children) {
        const childProb = computeNodeProb(child, 0.5);
        const childRho = child.rho ?? 0;
        const contribution = childRho * (childProb - 0.5) * k;
        logOdds += contribution;
      }

      const prob = 1 / (1 + Math.exp(-logOdds));
      const clampedProb = Math.max(0.01, Math.min(0.99, prob));
      probs[node.id] = clampedProb;
      return clampedProb;
    };

    if (state.data.tree.target) {
      computeNodeProb(state.data.tree.target, targetPrior);
    }

    return probs;
  }, [state.data, state.resolutions]);

  // Sensitivity analysis - compute marginal impact of each leaf
  const sensitivityAnalysis = useMemo(() => {
    if (!state.data) return [];

    const analysis: Array<{
      nodeId: string;
      text: string;
      currentProb: number;
      impactIfYes: number;
      impactIfNo: number;
      totalSwing: number;
    }> = [];

    const collectLeaves = (node: SignalNode): SignalNode[] => {
      if (!node.children?.length) return [node];
      return node.children.flatMap(collectLeaves);
    };

    const leaves = collectLeaves(state.data.tree.target);
    const currentTargetProb = computedProbabilities[state.data.tree.target.id] ??
      state.data.analysis.computed_probability;

    for (const leaf of leaves) {
      // Skip already resolved leaves
      if (state.resolutions[leaf.id]) continue;

      // Compute with YES
      const resolutionsWithYes = { ...state.resolutions, [leaf.id]: 'yes' as Resolution };
      const probWithYes = computeWithResolutions(state.data.tree.target, resolutionsWithYes,
        state.data.analysis.target_prior);

      // Compute with NO
      const resolutionsWithNo = { ...state.resolutions, [leaf.id]: 'no' as Resolution };
      const probWithNo = computeWithResolutions(state.data.tree.target, resolutionsWithNo,
        state.data.analysis.target_prior);

      analysis.push({
        nodeId: leaf.id,
        text: leaf.text,
        currentProb: leaf.base_rate ?? 0.5,
        impactIfYes: probWithYes - currentTargetProb,
        impactIfNo: probWithNo - currentTargetProb,
        totalSwing: Math.abs(probWithYes - probWithNo),
      });
    }

    // Sort by total swing (highest impact first)
    return analysis.sort((a, b) => b.totalSwing - a.totalSwing);
  }, [state.data, state.resolutions, computedProbabilities]);

  // Dynamic contributions (updates with resolutions)
  const dynamicContributions = useMemo((): ContributionData[] => {
    if (!state.data) return [];

    const collectLeaves = (node: SignalNode): SignalNode[] => {
      if (!node.children?.length) return [node];
      return node.children.flatMap(collectLeaves);
    };

    const leaves = collectLeaves(state.data.tree.target);
    const targetProb = computedProbabilities[state.data.tree.target.id] ??
      state.data.analysis.computed_probability;
    const targetPrior = state.data.analysis.target_prior;

    return leaves.map(leaf => {
      const prob = computedProbabilities[leaf.id] ?? leaf.base_rate ?? 0.5;
      const rho = leaf.rho ?? 0;
      const contribution = Math.abs(rho * (prob - 0.5) * 2);

      return {
        signal_id: leaf.id,
        text: leaf.text,
        base_rate: prob,
        rho,
        contribution,
        direction: (rho > 0 ? 'enhances' : 'suppresses') as 'enhances' | 'suppresses',
        spread: Math.abs((leaf.p_parent_given_yes ?? 0.5) - (leaf.p_parent_given_no ?? 0.5)),
        p_parent_given_yes: leaf.p_parent_given_yes ?? 0.5,
        p_parent_given_no: leaf.p_parent_given_no ?? 0.5,
      };
    }).sort((a, b) => b.contribution - a.contribution);
  }, [state.data, computedProbabilities]);

  // Helper to compute with specific resolutions
  function computeWithResolutions(
    node: SignalNode,
    resolutions: Record<string, Resolution>,
    targetPrior: number
  ): number {
    const computeNodeProb = (n: SignalNode, prior: number = 0.5): number => {
      const resolution = resolutions[n.id];
      if (resolution === 'yes') return 1.0;
      if (resolution === 'no') return 0.0;

      if (!n.children || n.children.length === 0) {
        return n.base_rate ?? prior;
      }

      const safePrior = Math.max(0.01, Math.min(0.99, prior));
      let logOdds = Math.log(safePrior / (1 - safePrior));
      const k = 2.0;

      for (const child of n.children) {
        const childProb = computeNodeProb(child, 0.5);
        const childRho = child.rho ?? 0;
        logOdds += childRho * (childProb - 0.5) * k;
      }

      return Math.max(0.01, Math.min(0.99, 1 / (1 + Math.exp(-logOdds))));
    };

    return computeNodeProb(node, targetPrior);
  }

  // Helper to get display probability for a node
  const getNodeProbability = useCallback((nodeId: string): number | null => {
    const resolution = state.resolutions[nodeId];
    if (resolution === 'yes') return 1.0;
    if (resolution === 'no') return 0.0;

    if (Object.keys(state.resolutions).length > 0 && computedProbabilities[nodeId] != null) {
      return computedProbabilities[nodeId];
    }

    const node = findNode(state.data?.tree.target ?? null, nodeId);
    return node?.base_rate ?? null;
  }, [state.resolutions, computedProbabilities, state.data]);

  // Find a node by ID in the tree
  const findNode = useCallback((root: SignalNode | null, targetId: string): SignalNode | null => {
    if (!root) return null;
    if (root.id === targetId) return root;
    for (const child of root.children || []) {
      const found = findNode(child, targetId);
      if (found) return found;
    }
    return null;
  }, []);

  // Get all nodes matching search/filter
  const filteredNodes = useMemo(() => {
    if (!state.data) return new Set<string>();

    const matches = new Set<string>();
    const { searchQuery, probabilityMin, probabilityMax, showEnhancersOnly, showSuppressorsOnly } =
      state.filterSettings;

    const checkNode = (node: SignalNode) => {
      let isMatch = true;

      // Search query
      if (searchQuery && !node.text.toLowerCase().includes(searchQuery.toLowerCase())) {
        isMatch = false;
      }

      // Probability filter
      const prob = (computedProbabilities[node.id] ?? node.base_rate ?? 0.5) * 100;
      if (prob < probabilityMin || prob > probabilityMax) {
        isMatch = false;
      }

      // Direction filter
      if (showEnhancersOnly && (node.rho ?? 0) <= 0) isMatch = false;
      if (showSuppressorsOnly && (node.rho ?? 0) >= 0) isMatch = false;

      if (isMatch) matches.add(node.id);
      node.children?.forEach(checkNode);
    };

    checkNode(state.data.tree.target);
    return matches;
  }, [state.data, state.filterSettings, computedProbabilities]);

  // Get selected node
  const selectedNode = useMemo(() => {
    if (!state.selectedNodeId || !state.data) return null;
    return findNode(state.data.tree.target, state.selectedNodeId);
  }, [state.selectedNodeId, state.data, findNode]);

  // Get path from node to root
  const getPathToRoot = useCallback((nodeId: string): SignalNode[] => {
    if (!state.data) return [];

    const path: SignalNode[] = [];
    const findPath = (node: SignalNode, targetId: string, currentPath: SignalNode[]): boolean => {
      currentPath.push(node);
      if (node.id === targetId) {
        path.push(...currentPath);
        return true;
      }
      for (const child of node.children || []) {
        if (findPath(child, targetId, [...currentPath])) return true;
      }
      return false;
    };

    findPath(state.data.tree.target, nodeId, []);
    return path.reverse();
  }, [state.data]);

  // Get all leaves
  const getAllLeaves = useCallback((): SignalNode[] => {
    if (!state.data) return [];
    const leaves: SignalNode[] = [];
    const collect = (node: SignalNode) => {
      if (!node.children?.length) leaves.push(node);
      else node.children.forEach(collect);
    };
    collect(state.data.tree.target);
    return leaves;
  }, [state.data]);

  // Computed stats
  const stats = useMemo(() => {
    if (!state.data) return null;

    const analysis = state.data.analysis;
    const hasResolutions = Object.keys(state.resolutions).length > 0;
    const originalProb = analysis?.computed_probability ?? state.data.tree.computed_probability;
    const currentProb = hasResolutions
      ? computedProbabilities[state.data.tree.target.id]
      : originalProb;

    return {
      totalSignals: analysis?.total_signals ?? state.data.tree.signals?.length ?? 0,
      leafCount: analysis?.leaf_count ?? state.data.tree.leaf_count ?? 0,
      maxDepth: analysis?.max_depth ?? state.data.tree.max_depth ?? 0,
      originalProbability: originalProb,
      currentProbability: currentProb,
      hasResolutions,
      resolutionCount: Object.keys(state.resolutions).length,
    };
  }, [state.data, state.resolutions, computedProbabilities]);

  return {
    // State
    data: state.data,
    selectedNodeId: state.selectedNodeId,
    resolutions: state.resolutions,
    computedProbabilities,
    selectedNode,
    stats,
    collapsedNodes: state.collapsedNodes,
    scenarios: state.scenarios,
    activeScenarioId: state.activeScenarioId,
    viewSettings: state.viewSettings,
    filterSettings: state.filterSettings,
    highlightedPath: state.highlightedPath,
    batchSelectMode: state.batchSelectMode,
    batchSelectedNodes: state.batchSelectedNodes,
    filteredNodes,
    sensitivityAnalysis,
    dynamicContributions,

    // Data actions
    loadData,
    selectNode,
    resolveSignal,
    resetAllResolutions,

    // Collapse actions
    toggleCollapse,
    expandAll,
    collapseAll,

    // View actions
    setZoom,
    setPan,
    resetView,
    toggleDarkMode,
    toggleMinimap,
    toggleDetailPanel,

    // Filter actions
    setSearchQuery,
    setProbabilityFilter,
    setDirectionFilter,
    clearFilters,

    // Scenario actions
    saveScenario,
    loadScenario,
    deleteScenario,

    // Batch actions
    toggleBatchMode,
    toggleBatchSelect,
    batchResolve,

    // Path highlight
    setHighlightedPath,
    clearHighlightedPath,

    // Helpers
    getNodeProbability,
    findNode: (nodeId: string) => findNode(state.data?.tree.target ?? null, nodeId),
    getPathToRoot,
    getAllLeaves,
  };
}

export type TreeStateHook = ReturnType<typeof useTreeState>;
