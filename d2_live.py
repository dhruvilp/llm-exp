import fs from 'node:fs';
import path from 'node:path';
import { D2 } from '@terrastruct/d2';

const INPUT_FILE = './input.d2';
const OUTPUT_FILE = './preview.svg';

const d2 = new D2();

// Core rendering function
async function compileD2() {
  try {
    console.log(`[${new Date().toLocaleTimeString()}] Compiling ${INPUT_FILE}...`);
    
    const d2Code = fs.readFileSync(INPUT_FILE, 'utf-8');
    const result = await d2.compile(d2Code);
    const svgOutput = await d2.render(result.diagram, result.renderOptions);
    
    fs.writeFileSync(OUTPUT_FILE, svgOutput);
    console.log(`✅ Preview updated successfully at ${OUTPUT_FILE}`);
  } catch (error) {
    console.error('❌ Compilation failed:', error.message);
  }
}

// 1. Run an immediate initial compilation when the script starts
await compileD2();

console.log(`\n👀 Watching for changes on ${INPUT_FILE}... Press Ctrl+C to stop.\n`);

// 2. Setup the file watcher with a debounce to prevent double-firing
let fsWait = false;
fs.watch(INPUT_FILE, async (eventType) => {
  if (eventType === 'change') {
    if (fsWait) return;
    
    // Lock the watcher for 100ms
    fsWait = setTimeout(() => {
      fsWait = false;
    }, 100);

    await compileD2();
  }
});

"""
How to run itMake sure your package.json includes "type": "module" to support ES imports.
Ensure your input.d2 file exists in the same folder.
Start your script in the terminal: `node watch-d2.js`

"""
