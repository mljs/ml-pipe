import { RandomForestRegressor } from '../../estimators/ensemble/randomForestRegressor';
import { LinearRegressor } from '../../estimators/linear/linearRegressor';
import { trainingSet, labels } from '../../utils/testHelpers';
import { generalizedDegreesOfFreedom } from '../complexity';

describe('test generalized degrees of freedom', () => {
  it('on linear regressor', async () => {
    const regressor = new LinearRegressor();
    await regressor.fit(trainingSet, labels);
    const gdf = await generalizedDegreesOfFreedom(
      trainingSet,
      labels,
      regressor,
    );
    expect(gdf).toBeGreaterThanOrEqual(1);
  });

  it('on random forest regressor', async () => {
    const regressor = new RandomForestRegressor({
      seed: 3,
      maxFeatures: 2,
      replacement: false,
      nEstimators: 200,
      treeOptions: undefined,
      useSampleBagging: true,
    });
    await regressor.fit(trainingSet, labels);
    const gdf = await generalizedDegreesOfFreedom(
      trainingSet,
      labels,
      regressor,
    );
    expect(gdf).toBeGreaterThanOrEqual(9);
  });
});
