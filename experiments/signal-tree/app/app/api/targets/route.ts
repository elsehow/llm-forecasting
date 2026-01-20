import { NextResponse } from 'next/server';
import { readdir, stat, readFile } from 'fs/promises';
import path from 'path';

export interface TreeFile {
  id: string;
  label: string;
  path: string;
  filename: string;
  timestamp: string;
  mtime: number;
  targetQuestion: string;
  computedProbability: number | null;
  marketPrice: number | null;
}

export interface TargetFolder {
  slug: string;
  label: string;
  trees: TreeFile[];
  latestTree: TreeFile | null;
}

async function findTreeFiles(dir: string, baseDir: string): Promise<TreeFile[]> {
  const files: TreeFile[] = [];

  try {
    const entries = await readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        // Recurse into subdirectories
        const subFiles = await findTreeFiles(fullPath, baseDir);
        files.push(...subFiles);
      } else if (entry.name.endsWith('.json') && entry.name.startsWith('tree_')) {
        // Found a tree file
        const stats = await stat(fullPath);
        const relativePath = path.relative(baseDir, fullPath);

        // Extract timestamp from filename (e.g., tree_20260119_cc.json -> 20260119)
        const timestampMatch = entry.name.match(/(\d{8})/);
        const timestamp = timestampMatch ? timestampMatch[1] : '';

        // Read JSON to get target question and computed probability
        let targetQuestion = '';
        let computedProbability: number | null = null;
        let marketPrice: number | null = null;

        try {
          const content = await readFile(fullPath, 'utf-8');
          const data = JSON.parse(content);
          targetQuestion = data.target?.text || data.tree?.target?.text || '';
          computedProbability = data.computed_probability ?? null;
          marketPrice = data.target?.market_price ?? data.market_validation?.market_price ?? null;
        } catch {
          // If parsing fails, continue with defaults
        }

        // Create readable label from target question or filename
        const label = targetQuestion
          ? truncateLabel(targetQuestion, 50)
          : entry.name.replace('.json', '').replace(/_/g, ' ');

        files.push({
          id: relativePath.replace(/\//g, '_').replace('.json', ''),
          label,
          path: `/api/tree?file=${encodeURIComponent(relativePath)}`,
          filename: entry.name,
          timestamp,
          mtime: stats.mtimeMs,
          targetQuestion,
          computedProbability,
          marketPrice,
        });
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${dir}:`, error);
  }

  return files;
}

export async function GET() {
  // Look in results directory (parent of app)
  const resultsDir = path.join(process.cwd(), '..', 'results');

  try {
    const allTrees = await findTreeFiles(resultsDir, resultsDir);

    // Group by parent folder (target slug)
    const folderMap = new Map<string, TreeFile[]>();

    for (const tree of allTrees) {
      // Extract folder from path (e.g., "/api/tree?file=one_battle_best_picture%2Ftree_xxx.json")
      const fileParam = tree.path.split('file=')[1] || '';
      const decodedPath = decodeURIComponent(fileParam);
      const parts = decodedPath.split('/');
      const folder = parts.length > 1 ? parts[0] : 'root';

      if (!folderMap.has(folder)) {
        folderMap.set(folder, []);
      }
      folderMap.get(folder)!.push(tree);
    }

    // Convert to TargetFolder array
    const targets: TargetFolder[] = [];

    for (const [slug, trees] of folderMap) {
      // Sort trees by mtime (most recent first)
      trees.sort((a, b) => b.mtime - a.mtime);

      // Create folder label from slug
      const label = slug
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

      targets.push({
        slug,
        label,
        trees,
        latestTree: trees[0] || null,
      });
    }

    // Sort folders by latest tree mtime
    targets.sort((a, b) => {
      const aTime = a.latestTree?.mtime ?? 0;
      const bTime = b.latestTree?.mtime ?? 0;
      return bTime - aTime;
    });

    return NextResponse.json(targets);
  } catch (error) {
    console.error('Error reading results directory:', error);
    return NextResponse.json([], { status: 500 });
  }
}

function truncateLabel(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + '…';
}
