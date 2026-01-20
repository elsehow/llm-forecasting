'use client';

import { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import * as d3 from 'd3';
import type { SignalNode, Resolution } from '@/lib/types';

interface TreeCanvasProps {
  root: SignalNode;
  selectedNodeId: string | null;
  resolutions: Record<string, Resolution>;
  computedProbabilities: Record<string, number>;
  collapsedNodes: Set<string>;
  filteredNodes: Set<string>;
  highlightedPath: string[];
  batchSelectMode: boolean;
  batchSelectedNodes: Set<string>;
  zoom: number;
  panX: number;
  panY: number;
  showMinimap: boolean;
  onNodeClick: (nodeId: string) => void;
  onNodeDoubleClick: (nodeId: string) => void;
  onBatchSelect: (nodeId: string) => void;
  onZoomChange: (zoom: number) => void;
  onPanChange: (x: number, y: number) => void;
}

interface D3Node extends d3.HierarchyPointNode<SignalNode> {}

export function TreeCanvas({
  root,
  selectedNodeId,
  resolutions,
  computedProbabilities,
  collapsedNodes,
  filteredNodes,
  highlightedPath,
  batchSelectMode,
  batchSelectedNodes,
  zoom,
  panX,
  panY,
  showMinimap,
  onNodeClick,
  onNodeDoubleClick,
  onBatchSelect,
  onZoomChange,
  onPanChange,
}: TreeCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isMinimapDragging, setIsMinimapDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Build tree with collapsed nodes filtered out
  const visibleRoot = useMemo(() => {
    const filterCollapsed = (node: SignalNode): SignalNode => {
      if (collapsedNodes.has(node.id)) {
        return { ...node, children: [] };
      }
      return {
        ...node,
        children: node.children?.map(filterCollapsed) || [],
      };
    };
    return filterCollapsed(root);
  }, [root, collapsedNodes]);

  // Build D3 hierarchy
  const { nodes, links, width, height } = useMemo(() => {
    const hierarchy = d3.hierarchy(visibleRoot);
    const leafCount = hierarchy.leaves().length;
    const maxDepth = hierarchy.height;

    const nodeWidth = 180;
    const horizontalSpacing = Math.max(nodeWidth + 40, 220);
    const verticalSpacing = 160;

    const innerWidth = Math.max(leafCount * horizontalSpacing, 800);
    const innerHeight = (maxDepth + 1) * verticalSpacing;

    const margin = { top: 80, right: 100, bottom: 80, left: 100 };
    const width = innerWidth + margin.left + margin.right;
    const height = innerHeight + margin.top + margin.bottom;

    const treeLayout = d3.tree<SignalNode>()
      .size([innerWidth, innerHeight])
      .separation((a, b) => (a.parent === b.parent ? 1.2 : 1.8));

    const treeRoot = treeLayout(hierarchy);

    treeRoot.each(node => {
      node.x += margin.left;
      node.y += margin.top;
    });

    return {
      nodes: treeRoot.descendants(),
      links: treeRoot.links(),
      width,
      height,
    };
  }, [visibleRoot]);

  // Get path to root for highlighting
  const pathNodeIds = useMemo(() => {
    if (!selectedNodeId && highlightedPath.length === 0) return new Set<string>();

    const ids = new Set<string>(highlightedPath);

    if (selectedNodeId) {
      const findPath = (node: SignalNode, targetId: string, path: string[]): boolean => {
        path.push(node.id);
        if (node.id === targetId) {
          path.forEach(id => ids.add(id));
          return true;
        }
        for (const child of node.children || []) {
          if (findPath(child, targetId, [...path])) return true;
        }
        return false;
      };
      findPath(root, selectedNodeId, []);
    }

    return ids;
  }, [root, selectedNodeId, highlightedPath]);

  // Get probability for display
  const getProb = (nodeId: string, baseRate: number | null): string => {
    const resolution = resolutions[nodeId];
    if (resolution === 'yes') return '100%';
    if (resolution === 'no') return '0%';

    const computed = computedProbabilities[nodeId];
    const prob = computed ?? baseRate;
    return prob != null ? `${(prob * 100).toFixed(0)}%` : '-';
  };

  // Node styling based on state
  const getNodeClasses = (node: D3Node): string => {
    const classes = ['node-bg'];

    // Type
    if (node.depth === 0) classes.push('root');
    else if (!node.data.children?.length) classes.push('leaf');
    else classes.push('internal');

    // Has collapsed children
    if (collapsedNodes.has(node.data.id)) classes.push('collapsed');

    // Resolution state
    const resolution = resolutions[node.data.id];
    if (resolution === 'yes') classes.push('resolved-yes');
    else if (resolution === 'no') classes.push('resolved-no');

    // Selection state
    if (selectedNodeId) {
      if (node.data.id === selectedNodeId) classes.push('selected');
      else if (!pathNodeIds.has(node.data.id)) classes.push('dimmed');
    }

    // Highlighted path
    if (highlightedPath.length > 0 && highlightedPath.includes(node.data.id)) {
      classes.push('path-highlighted');
    }

    // Batch selection
    if (batchSelectMode) {
      if (batchSelectedNodes.has(node.data.id)) classes.push('batch-selected');
    }

    // Filter matching
    if (filteredNodes.size > 0 && filteredNodes.size < nodes.length) {
      if (!filteredNodes.has(node.data.id)) classes.push('filtered-out');
    }

    // Hover
    if (hoveredNodeId === node.data.id) classes.push('hovered');

    return classes.join(' ');
  };

  // Link styling
  const getLinkClasses = (link: d3.HierarchyPointLink<SignalNode>): string => {
    const classes = ['link'];
    const rho = link.target.data.rho ?? 0;

    if (rho > 0) classes.push('enhances');
    else if (rho < 0) classes.push('suppresses');

    if (selectedNodeId || highlightedPath.length > 0) {
      if (pathNodeIds.has(link.source.data.id) && pathNodeIds.has(link.target.data.id)) {
        classes.push('highlighted');
      } else {
        classes.push('dimmed');
      }
    }

    return classes.join(' ');
  };

  // Generate curved path
  const linkPath = (link: d3.HierarchyPointLink<SignalNode>): string => {
    const source = link.source;
    const target = link.target;
    return `M ${source.x} ${source.y + 30}
            C ${source.x} ${(source.y + target.y) / 2},
              ${target.x} ${(source.y + target.y) / 2},
              ${target.x} ${target.y - 30}`;
  };

  // Pan handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (e.button === 0 && !target.closest?.('.node-group') && !target.closest?.('.minimap')) {
      e.preventDefault(); // Prevent text selection
      setIsDragging(true);
      setDragStart({ x: e.clientX - panX, y: e.clientY - panY });
    }
  }, [panX, panY]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      onPanChange(e.clientX - dragStart.x, e.clientY - dragStart.y);
    }
  }, [isDragging, dragStart, onPanChange]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setIsMinimapDragging(false);
  }, []);

  // Minimap handlers for navigation
  const handleMinimapInteraction = useCallback((e: React.MouseEvent) => {
    const minimap = minimapRef.current;
    if (!minimap || !containerRef.current) return;

    const rect = minimap.getBoundingClientRect();
    const minimapWidth = rect.width;
    const minimapHeight = rect.height;

    // Calculate where user clicked in minimap coordinates (0-1)
    const clickX = (e.clientX - rect.left) / minimapWidth;
    const clickY = (e.clientY - rect.top) / minimapHeight;

    // Convert to tree coordinates
    const treeX = clickX * width;
    const treeY = clickY * height;

    // Calculate pan to center viewport on clicked point
    const viewportWidth = containerRef.current.clientWidth;
    const viewportHeight = containerRef.current.clientHeight;

    const newPanX = -(treeX * zoom - viewportWidth / 2);
    const newPanY = -(treeY * zoom - viewportHeight / 2);

    onPanChange(newPanX, newPanY);
  }, [width, height, zoom, onPanChange]);

  const handleMinimapMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsMinimapDragging(true);
    handleMinimapInteraction(e);
  }, [handleMinimapInteraction]);

  const handleMinimapMouseMove = useCallback((e: React.MouseEvent) => {
    if (isMinimapDragging) {
      handleMinimapInteraction(e);
    }
  }, [isMinimapDragging, handleMinimapInteraction]);

  // Wheel handler: scroll = pan, Ctrl+scroll = zoom
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      if (e.ctrlKey || e.metaKey) {
        // Ctrl/Cmd + scroll = zoom
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        onZoomChange(Math.max(0.25, Math.min(2, zoom + delta)));
      } else {
        // Regular scroll = pan
        onPanChange(panX - e.deltaX, panY - e.deltaY);
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [zoom, panX, panY, onZoomChange, onPanChange]);

  // Handle node click
  const handleNodeClick = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    if (batchSelectMode) {
      onBatchSelect(nodeId);
    } else {
      onNodeClick(nodeId);
    }
  }, [batchSelectMode, onNodeClick, onBatchSelect]);

  // Handle double-click (collapse/expand)
  const handleNodeDoubleClick = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    onNodeDoubleClick(nodeId);
  }, [onNodeDoubleClick]);

  // Check if node has children (including collapsed ones)
  const hasChildren = (nodeId: string): boolean => {
    const findNode = (node: SignalNode): SignalNode | null => {
      if (node.id === nodeId) return node;
      for (const child of node.children || []) {
        const found = findNode(child);
        if (found) return found;
      }
      return null;
    };
    const node = findNode(root);
    return (node?.children?.length ?? 0) > 0;
  };

  return (
    <div
      ref={containerRef}
      className={`tree-canvas ${isDragging ? 'dragging' : ''}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Zoom controls */}
      <div className="zoom-controls">
        <button onClick={() => onZoomChange(Math.min(2, zoom + 0.25))} title="Zoom in">+</button>
        <span className="zoom-level">{Math.round(zoom * 100)}%</span>
        <button onClick={() => onZoomChange(Math.max(0.25, zoom - 0.25))} title="Zoom out">−</button>
        <button onClick={() => { onZoomChange(1); onPanChange(0, 0); }} title="Reset view">⟲</button>
      </div>

      {/* Main SVG */}
      <svg
        width={width * zoom}
        height={height * zoom}
        className="tree-svg"
        style={{
          transform: `translate(${panX}px, ${panY}px)`,
        }}
      >
        <g transform={`scale(${zoom})`}>
          {/* Links */}
          <g className="links">
            {links.map((link, i) => (
              <g key={i}>
                <path
                  d={linkPath(link)}
                  className={getLinkClasses(link)}
                  strokeWidth={1.5 + Math.abs(link.target.data.rho ?? 0) * 3}
                />
                {/* Rho label */}
                {link.target.data.rho != null && (
                  <text
                    x={(link.source.x + link.target.x) / 2}
                    y={(link.source.y + link.target.y) / 2 - 8}
                    className={`link-label ${(link.target.data.rho ?? 0) > 0 ? 'enhances' : 'suppresses'}`}
                    textAnchor="middle"
                  >
                    ρ={link.target.data.rho > 0 ? '+' : ''}{link.target.data.rho.toFixed(2)}
                  </text>
                )}
              </g>
            ))}
          </g>

          {/* Nodes */}
          <g className="nodes">
            {nodes.map(node => (
              <g
                key={node.data.id}
                className="node-group"
                transform={`translate(${node.x}, ${node.y})`}
                onClick={(e) => handleNodeClick(e, node.data.id)}
                onDoubleClick={(e) => handleNodeDoubleClick(e, node.data.id)}
                onMouseEnter={() => setHoveredNodeId(node.data.id)}
                onMouseLeave={() => setHoveredNodeId(null)}
              >
                {/* Background rect */}
                <rect
                  className={getNodeClasses(node)}
                  x={-90}
                  y={-30}
                  width={180}
                  height={60}
                  rx={8}
                />

                {/* Collapse indicator */}
                {hasChildren(node.data.id) && (
                  <circle
                    className={`collapse-indicator ${collapsedNodes.has(node.data.id) ? 'collapsed' : ''}`}
                    cx={75}
                    cy={20}
                    r={10}
                  >
                    <title>Double-click to {collapsedNodes.has(node.data.id) ? 'expand' : 'collapse'}</title>
                  </circle>
                )}
                {hasChildren(node.data.id) && (
                  <text
                    className="collapse-icon"
                    x={75}
                    y={24}
                    textAnchor="middle"
                  >
                    {collapsedNodes.has(node.data.id) ? '+' : '−'}
                  </text>
                )}

                {/* Resolution icon */}
                {resolutions[node.data.id] && (
                  <text
                    className={`resolution-icon ${resolutions[node.data.id]}`}
                    x={-75}
                    y={-15}
                    textAnchor="middle"
                  >
                    {resolutions[node.data.id] === 'yes' ? '✓' : '✗'}
                  </text>
                )}

                {/* Batch select checkbox */}
                {batchSelectMode && !node.data.children?.length && (
                  <rect
                    className={`batch-checkbox ${batchSelectedNodes.has(node.data.id) ? 'checked' : ''}`}
                    x={-85}
                    y={-25}
                    width={16}
                    height={16}
                    rx={3}
                  />
                )}

                {/* Node text */}
                <text className="node-text" y={-8} textAnchor="middle">
                  {truncate(node.data.text, 26)}
                </text>

                {/* Probability */}
                <text className="node-prob" y={10} textAnchor="middle">
                  {getProb(node.data.id, node.data.base_rate)}
                </text>

                {/* Source badge */}
                {node.data.probability_source && (
                  <text
                    className={`node-source ${node.data.probability_source}`}
                    y={24}
                    textAnchor="middle"
                  >
                    {node.data.probability_source}
                  </text>
                )}

                {/* Necessity badge */}
                {node.data.relationship_type === 'necessity' && (
                  <text
                    className="node-necessity"
                    y={node.data.probability_source ? 38 : 24}
                    textAnchor="middle"
                  >
                    🔒 REQUIRED
                  </text>
                )}

                {/* Market indicator */}
                {node.data.market_price != null && (
                  <g className="market-indicator">
                    <circle
                      cx={80}
                      cy={-20}
                      r={8}
                      className="market-indicator-bg"
                    />
                    <text
                      x={80}
                      y={-16}
                      textAnchor="middle"
                      className="market-indicator-icon"
                    >
                      📊
                    </text>
                  </g>
                )}

                {/* Tooltip */}
                <title>
                  {node.data.text}
                  {'\n\n'}P(YES): {getProb(node.data.id, node.data.base_rate)}
                  {node.data.rho != null && `\nρ = ${node.data.rho > 0 ? '+' : ''}${node.data.rho.toFixed(2)}`}
                  {node.data.resolution_date && `\nResolves: ${new Date(node.data.resolution_date).toLocaleDateString()}`}
                  {hasChildren(node.data.id) && '\n\nDouble-click to collapse/expand'}
                </title>
              </g>
            ))}
          </g>
        </g>
      </svg>

      {/* Minimap */}
      {showMinimap && (
        <div
          ref={minimapRef}
          className={`minimap ${isMinimapDragging ? 'dragging' : ''}`}
          onMouseDown={handleMinimapMouseDown}
          onMouseMove={handleMinimapMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <svg
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Links */}
            {links.map((link, i) => (
              <path
                key={i}
                d={linkPath(link)}
                className={`minimap-link ${(link.target.data.rho ?? 0) > 0 ? 'enhances' : 'suppresses'}`}
              />
            ))}
            {/* Nodes */}
            {nodes.map(node => (
              <circle
                key={node.data.id}
                cx={node.x}
                cy={node.y}
                r={8}
                className={`minimap-node ${
                  node.data.id === selectedNodeId ? 'selected' : ''
                } ${resolutions[node.data.id] ? `resolved-${resolutions[node.data.id]}` : ''}`}
              />
            ))}
            {/* Viewport indicator */}
            <rect
              className="minimap-viewport"
              x={-panX / zoom}
              y={-panY / zoom}
              width={(containerRef.current?.clientWidth ?? 400) / zoom}
              height={(containerRef.current?.clientHeight ?? 300) / zoom}
            />
          </svg>
        </div>
      )}

      {/* Legend */}
      <div className="legend">
        <div className="legend-row">
          <div className="legend-swatch enhances" />
          <span>Enhances (ρ &gt; 0)</span>
        </div>
        <div className="legend-row">
          <div className="legend-swatch suppresses" />
          <span>Suppresses (ρ &lt; 0)</span>
        </div>
        <div className="legend-row">
          <div className="legend-node leaf" />
          <span>Leaf (actionable)</span>
        </div>
        <div className="legend-row">
          <div className="legend-node internal" />
          <span>Internal</span>
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredNodeId && (
        <HoverTooltip
          node={nodes.find(n => n.data.id === hoveredNodeId)?.data}
          prob={getProb(hoveredNodeId, nodes.find(n => n.data.id === hoveredNodeId)?.data.base_rate ?? null)}
        />
      )}
    </div>
  );
}

// Hover tooltip component
function HoverTooltip({ node, prob }: { node?: SignalNode; prob: string }) {
  if (!node) return null;

  return (
    <div className="hover-tooltip">
      <div className="tooltip-text">{node.text}</div>
      <div className="tooltip-meta">
        <span>P(YES): {prob}</span>
        {node.rho != null && (
          <span className={node.rho > 0 ? 'enhances' : 'suppresses'}>
            ρ = {node.rho > 0 ? '+' : ''}{node.rho.toFixed(2)}
          </span>
        )}
      </div>
      {node.resolution_date && (
        <div className="tooltip-date">
          Resolves: {new Date(node.resolution_date).toLocaleDateString()}
        </div>
      )}
    </div>
  );
}

function truncate(text: string, maxLen: number): string {
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '…' : text;
}
