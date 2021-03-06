import seedrandom from 'seedrandom';

import { RandomForestRegressor } from '../../estimators/ensemble/RandomForestRegressor';
import { LinearRegressor } from '../../estimators/linear/LinearRegressor';
import { trainingSet, labels } from '../../utils/testHelpers';
import { generalizedDegreesOfFreedom } from '../complexity';

describe('test generalized degrees of freedom', () => {
  it('on linear regressor', async () => {
    seedrandom('test');
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
    seedrandom('test');
    const regressor = new RandomForestRegressor();
    await regressor.fit(trainingSet, labels);
    const gdf = await generalizedDegreesOfFreedom(
      trainingSet,
      labels,
      regressor,
    );
    expect(gdf).toBeGreaterThanOrEqual(9);
  });
});
