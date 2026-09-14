const test = require('node:test');
const assert = require('node:assert/strict');
const { prepareBoxedDownload } = require('../static/app.js');

test('starts once then polls until the ready attachment URL', async () => {
  const calls = [];
  const states = [];
  const responses = [{state: 'queued'}, {state: 'running'}, {state: 'ready', download_url: '/api/events/a/download/file'}];
  const result = await prepareBoxedDownload('/api/events/a/download', {
    request: async (url, method) => { calls.push([url, method]); return responses.shift(); },
    onState: state => states.push(state), isCurrent: () => true,
    wait: async () => {},
  });
  assert.equal(result, '/api/events/a/download/file');
  assert.deepEqual(calls, [['/api/events/a/download', 'POST'], ['/api/events/a/download', 'GET'], ['/api/events/a/download', 'GET']]);
  assert.deepEqual(states, ['queued', 'running', 'ready']);
});

test('switching events while export completes must not download the old event', async () => {
  let current = true;
  const result = await prepareBoxedDownload('/api/runs/live_old/events/a/download', {
    request: async () => { current = false; return {state: 'ready', download_url: '/old/file'}; },
    onState: () => assert.fail('stale export changed current button'),
    isCurrent: () => current, wait: async () => {},
  });
  assert.equal(result, null);
});

test('failed generation and exhausted polling remain retryable errors', async () => {
  await assert.rejects(prepareBoxedDownload('/api/events/a/download', {
    request: async () => ({state: 'error', error: 'generation failed'}),
    onState: () => {}, isCurrent: () => true, wait: async () => {},
  }), /generation failed/);
  let count = 0;
  await assert.rejects(prepareBoxedDownload('/api/events/a/download', {
    request: async () => { count++; return {state: 'queued'}; },
    onState: () => {}, isCurrent: () => true, wait: async () => {},
  }), /稍后重试/);
  assert.equal(count, 121);
});

test('rejects an unexpected cross-origin download destination', async () => {
  await assert.rejects(prepareBoxedDownload('/api/events/a/download', {
    request: async () => ({state: 'ready', download_url: 'https://other.example/video'}),
    onState: () => {}, isCurrent: () => true, wait: async () => {},
  }), /下载地址/);
});
