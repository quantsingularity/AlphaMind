module.exports = (api) => {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
    plugins: [],
    env: {
      test: {
        plugins: ["@babel/plugin-transform-flow-strip-types"],
      },
    },
  };
};
