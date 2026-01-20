import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import path from 'path';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const file = searchParams.get('file');

  if (!file) {
    return NextResponse.json({ error: 'Missing file parameter' }, { status: 400 });
  }

  // Prevent directory traversal
  const normalized = path.normalize(file);
  if (normalized.includes('..')) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  const resultsDir = path.join(process.cwd(), '..', 'results');
  const fullPath = path.join(resultsDir, normalized);

  // Ensure the path is within results directory
  if (!fullPath.startsWith(resultsDir)) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  try {
    const content = await readFile(fullPath, 'utf-8');
    const data = JSON.parse(content);
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Error reading tree file ${fullPath}:`, error);
    return NextResponse.json({ error: 'File not found' }, { status: 404 });
  }
}
