import { Matrix } from 'ml-matrix';

import { Counter } from '../../../utils/array';
import { DummyClassifier } from '../DummyClassifier';

describe('DummyClassifier', () => {
  it('majority', async () => {
    const classifier = new DummyClassifier({ strategy: 'majority' });
    const X = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const y = new Matrix([[1], [1], [0]]);
    await classifier.fit(X, y);
    const prediction = classifier.predict(X);
    expect(prediction.to1DArray()).toStrictEqual([1, 1, 1]);
  });
  it('stratified', async () => {
    const classifier = new DummyClassifier({ strategy: 'stratified' });
    const X = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const y = new Matrix([[1], [1], [0]]);
    await classifier.fit(X, y);
    const newX = new Matrix(1000, 2).fill(0);
    const prediction = classifier.predict(newX);
    const counter = new Counter(prediction.to1DArray());
    expect(counter.get(1)).toBeGreaterThan(550);
    expect(counter.get(0)).toBeGreaterThan(250);
    expect(counter.get(0)).toBeLessThan(400);
    expect(counter.get(1)).toBeLessThan(700);
  });
});
