"use strict";

// Pure arithmetic regressions for the source-bound GAE translation. This file does
// not import the simulator, trainer, optimizer, or checkpoint deserializer.
const fs = require("node:fs");
const path = require("node:path");
const {runSyntheticGaeTests} = require("./r33_temporal_credit_audit.js");

const result = runSyntheticGaeTests();
if (result.failed !== 0 || result.passed !== result.tests_run) {
  throw new Error(`synthetic GAE tests failed: ${JSON.stringify(result)}`);
}
fs.writeFileSync(path.join(__dirname, "SYNTHETIC_GAE_TESTS.json"), JSON.stringify(result, null, 2) + "\n", "utf8");
console.log(JSON.stringify({tests_run: result.tests_run, passed: result.passed, failed: result.failed}, null, 2));
