module.exports = function (api) {
  api.cache(true);
  return {
    presets: [["babel-preset-expo", { jsxImportSource: "react" }], "@babel/preset-flow"],
    plugins: [],
    env: {
      test: {
        plugins: ["@babel/plugin-transform-flow-strip-types"],
      },
    },
  };
};
