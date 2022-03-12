import { sequential, layers } from '@tensorflow/tfjs';

function get_model(
  inputShape: number,
  hiddenShapes: number[],
  outputShape: number,
  hiddenActivation: 'relu' | 'sigmoid' | 'tanh' | 'linear',
  finalActivation: 'relu' | 'sigmoid' | 'tanh' | 'linear',
  kernelInitializer:
    | 'leCunNormal'
    | 'glorotUniform'
    | 'randomUniform'
    | 'truncatedNormal'
    | 'varianceScaling',
) {
  const model = sequential();
  model.add(
    layers.dense({
      inputShape: [inputShape],
      units: hiddenShapes[0],
      activation: hiddenActivation,
      kernelInitializer: kernelInitializer,
    }),
  );
  for (let i = 1; i < hiddenShapes.length; i++) {
    model.add(
      layers.dense({
        units: hiddenShapes[i],
        activation: hiddenActivation,
        kernelInitializer: kernelInitializer,
      }),
    );
  }
  model.add(
    layers.dense({
      units: outputShape,
      activation: finalActivation,
      kernelInitializer: kernelInitializer,
    }),
  );
  return model;
}
