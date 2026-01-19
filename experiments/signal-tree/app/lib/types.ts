// Signal Tree Types (mirrors Python models in shared/tree.py)

export type ProbabilitySource = 'polymarket' | 'metaculus' | 'llm' | 'manual';

export interface SignalNode {
  id: string;
  text: string;
  resolution_date: string | null;
  base_rate: number | null;
  probability_source: ProbabilitySource | null;
  parent_id: string | null;
  children: SignalNode[];
  rho: number | null;
  rho_reasoning: string | null;
  p_parent_given_yes: number | null;
  p_parent_given_no: number | null;
  is_leaf: boolean;
  depth: number;
}

export interface SignalTree {
  target: SignalNode;
  signals: SignalNode[];
  max_depth: number;
  leaf_count: number;
  actionable_horizon_days: number;
  computed_probability: number | null;
}

export interface TreeConfig {
  target_question: string;
  target_id: string;
  minimum_resolution_days: number;
  max_signals: number;
  actionable_horizon_days: number;
  signals_per_node: number;
  generation_model: string;
  rho_model: string;
  include_market_signals: boolean;
  market_match_threshold: number;
}

export interface ContributionData {
  signal_id: string;
  text: string;
  base_rate: number;
  rho: number;
  contribution: number;
  direction: 'enhances' | 'suppresses';
  spread: number;
  p_parent_given_yes: number;
  p_parent_given_no: number;
}

export interface TreeAnalysis {
  target: string;
  computed_probability: number;
  target_prior: number;
  max_depth: number;
  total_signals: number;
  leaf_count: number;
  top_contributors: ContributionData[];
  all_contributions: ContributionData[];
}

export interface TreeDataFile {
  config: TreeConfig;
  tree: SignalTree;
  analysis: TreeAnalysis;
  generated_at: string;
}

// UI State types
export type Resolution = 'yes' | 'no';

export interface Scenario {
  id: string;
  name: string;
  resolutions: Record<string, Resolution>;
  createdAt: number;
}

export interface ViewSettings {
  zoom: number;
  panX: number;
  panY: number;
  darkMode: boolean;
  showMinimap: boolean;
  detailPanelOpen: boolean;
}

export interface FilterSettings {
  searchQuery: string;
  probabilityMin: number;
  probabilityMax: number;
  showEnhancersOnly: boolean;
  showSuppressorsOnly: boolean;
}

export interface TreeState {
  data: TreeDataFile | null;
  selectedNodeId: string | null;
  resolutions: Record<string, Resolution>;
  computedProbabilities: Record<string, number>;
  collapsedNodes: Set<string>;
  scenarios: Scenario[];
  activeScenarioId: string | null;
  viewSettings: ViewSettings;
  filterSettings: FilterSettings;
}
