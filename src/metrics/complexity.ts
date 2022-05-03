import { Matrix } from 'ml-matrix';

import { Estimator } from '../estimators/estimator';
import { Pipeline } from '../pipeline/pipeline';
import { randomGaussianMatrix } from '../utils/array';
/**
 * Computes generalized degrees of freedom as originally proposed in [1].
 * Here, we use a Monte Carlo approximation, as described in [2]
 *
 *
 * [1] Ye, Jianming. “On Measuring and Correcting the Effects of Data Mining and Model Selection.” Journal of the American Statistical Association, vol. 93, no. 441, 1998, pp. 120–31, https://doi.org/10.2307/2669609. Accessed 3 May 2022.
 * [2] Gao, Tianxiang, και Vladimir Jojic. ‘Degrees of Freedom in Deep Neural Networks’. arXiv preprint arXiv: Arxiv-1603. 09260 (2016): n. pag. Print.
 * @export
 * @param {Matrix} X
 * @param {Matrix} Y
 * @param {(Estimator | Pipeline)} estimator
 * @param {1e-5} epsilon
 * @param {5} numberOfDraws
 * @return {*}  {Promise<number>}
 */
export async function generalizedDegreesOfFreedom(
  X: Matrix,
  Y: Matrix,
  estimator: Estimator | Pipeline,
  epsilon = 1e-3,
  numberOfDraws = 5,
): Promise<number> {
  const unperturbedPredictions = estimator.predict(X);
  let gdfEstimates = Matrix.zeros(1, numberOfDraws);
  for (let i = 0; i < numberOfDraws; i++) {
    const estimate = await gdfEstimate(
      X,
      Y,
      unperturbedPredictions,
      estimator,
      epsilon,
    );
    gdfEstimates.set(0, i, estimate);
  }
  return gdfEstimates.mean();
}

async function gdfEstimate(
  X: Matrix,
  Y: Matrix,
  unperturbedPredictions: Matrix,
  estimator: Estimator | Pipeline,
  epsilon: number,
): Promise<number> {
  const clonedEstimator = estimator; // structuredClone(estimator);
  const B = randomGaussianMatrix(Y.rows, Y.columns);
  const Ypert = Matrix.add(Y, Matrix.mul(B, epsilon));

  await clonedEstimator.fit(X, Ypert);

  const predictions = clonedEstimator.predict(X);

  let residuals = 0;
  for (let i = 0; i < B.rows; i++) {
    for (let j = 0; j < B.columns; j++) {
      residuals +=
        (B.get(i, j) *
          (predictions.get(i, j) - unperturbedPredictions.get(i, j))) /
        epsilon;
    }
  }
  return residuals;
}
