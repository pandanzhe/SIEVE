const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const root = path.resolve(__dirname, "..");
const input = path.join(root, "deliverables", "SIEVE-runtime-framework.svg");
const output = path.join(root, "deliverables", "SIEVE-runtime-framework.png");

async function main() {
  const svg = fs.readFileSync(input);
  await sharp(svg, { density: 180 })
    .png({ compressionLevel: 9 })
    .toFile(output);
  const metadata = await sharp(output).metadata();
  process.stdout.write(
    JSON.stringify({ output, width: metadata.width, height: metadata.height }) + "\n",
  );
}

main().catch((error) => {
  process.stderr.write(error.stack + "\n");
  process.exitCode = 1;
});
