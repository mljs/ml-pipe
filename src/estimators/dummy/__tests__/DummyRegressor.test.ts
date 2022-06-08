import { Matrix } from 'ml-matrix';

import { DummyRegressor } from '../DummyRegressor';

describe('DummyRegressor', () => {
  it('mean', async () => {
    const d = new DummyRegressor({ strategy: 'mean' });
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const y = new Matrix([[1], [2]]);
    await d.fit(X, y);
    expect(d.predict(X).to2DArray()).toStrictEqual([[1.5], [1.5]]);
  });
  it('constant', async () => {
    const d = new DummyRegressor({ strategy: 'constant', constantValue: 10 });
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const y = new Matrix([[1], [2]]);
    await d.fit(X, y);
    expect(d.predict(X).to2DArray()).toStrictEqual([[10], [10]]);
  });
  it('median', async () => {
    const d = new DummyRegressor({ strategy: 'median' });
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [4, 5, 6],
    ]);
    const y = new Matrix([[1], [2], [2]]);
    await d.fit(X, y);
    expect(d.predict(X).to2DArray()).toStrictEqual([[2], [2], [2]]);
  });
});
