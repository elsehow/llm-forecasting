import { NextResponse } from 'next/server';
import { readdir, stat, readFile } from 'fs/promises';
import path from 'path';

export interface TargetFile {
  id: string;
  label: string;
  path: string;
  timestamp: string;
  mtime: number;
  targetQuestion: string;
  generatedAt: string;
}

export async function GET() {
  const dataDir = path.join(process.cwd(), 'public', 'data');

  try {
    const files = await readdir(dataDir);
    const jsonFiles = files.filter(f => f.endsWith('.json'));

    const targets: TargetFile[] = await Promise.all(
      jsonFiles.map(async (filename) => {
        const filePath = path.join(dataDir, filename);
        const stats = await stat(filePath);

        // Extract timestamp from filename (e.g., tree_20260119.json -> 20260119)
        const timestampMatch = filename.match(/(\d{8})/);
        const timestamp = timestampMatch ? timestampMatch[1] : '';

        // Read JSON to get target question
        let targetQuestion = '';
        let generatedAt = '';
        try {
          const content = await readFile(filePath, 'utf-8');
          const data = JSON.parse(content);
          targetQuestion = data.config?.target_question || data.tree?.target?.text || '';
          generatedAt = data.generated_at || '';
        } catch {
          // If parsing fails, use filename
        }

        // Create readable label - prefer target question, fall back to filename
        const label = targetQuestion
          ? truncateLabel(targetQuestion, 60)
          : filename.replace('.json', '').replace(/_/g, ' ');

        return {
          id: filename.replace('.json', ''),
          label,
          path: `/data/${filename}`,
          timestamp,
          mtime: stats.mtimeMs,
          targetQuestion,
          generatedAt,
        };
      })
    );

    // Sort by modification time (most recent first)
    targets.sort((a, b) => b.mtime - a.mtime);

    return NextResponse.json(targets);
  } catch (error) {
    console.error('Error reading data directory:', error);
    return NextResponse.json([], { status: 500 });
  }
}

function truncateLabel(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + '…';
}
