const fs = require("fs");
const path = require("path");
const sharp = require(path.resolve(process.cwd(), "node_modules", "sharp"));

const outputDir = path.resolve(process.argv[2] || "outputs/paper/c1_manuscript_materials");
for (const filename of fs.readdirSync(outputDir).filter((name) => name.endsWith(".svg"))) {
  const input = path.join(outputDir, filename);
  const output = path.join(outputDir, filename.replace(/\.svg$/, ".png"));
  sharp(input, { density: 180 })
    .png({ compressionLevel: 9 })
    .toFile(output)
    .then(() => process.stdout.write(`${path.basename(output)}\n`))
    .catch((error) => {
      process.stderr.write(`${filename}: ${error.stack}\n`);
      process.exitCode = 1;
    });
}
