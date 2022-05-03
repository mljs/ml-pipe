import { Matrix } from 'ml-matrix';

import { Estimator } from '../estimators/estimator';
import { Pipeline } from '../pipeline/pipeline';

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
 * @param {number} epsilon
 * @param {5} numberOfDraws
 * @return {*}  {Promise<number>}
 */
export async function generalizedDegreesOfFreedom(
  X: Matrix,
  Y: Matrix,
  estimator: Estimator | Pipeline,
  epsilon: number,
  numberOfDraws: 5,
): Promise<number> {
  let gdfEstimates = Matrix.zeros(0, numberOfDraws);
  for (let i = 0; i < numberOfDraws; i++) {
    const estimate = await gdfEstimate(X, Y, estimator, epsilon);
    gdfEstimates.set(0, i, estimate);
  }
  return gdfEstimates.mean();
}

async function gdfEstimate(
  X: Matrix,
  Y: Matrix,
  estimator: Estimator | Pipeline,
  epsilon: number,
): Promise<number> {
  await estimator.fit(X, Y);

  const predictions = estimator.predict(X);
  let residuals = Y.sub(predictions);
  residuals = residuals.div(epsilon);
  const sumOfResiduals = residuals.sum();
  return sumOfResiduals;
}
