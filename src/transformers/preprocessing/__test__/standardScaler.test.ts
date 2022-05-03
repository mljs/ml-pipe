import { Matrix } from 'ml-matrix';

import { StandardScaler } from '../StandardScaler';

describe('test StandardScaler', () => {
  it('two cols', () => {
    const data = new Matrix([
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1],
    ]);

    const scaler = new StandardScaler();
    scaler.fit(data);
    const expected = new Matrix([
      [-1, -1],
      [-1, -1],
      [1, 1],
      [1, 1],
    ]);

    expect(scaler.transform(data)).toStrictEqual(expected);
    expect(scaler.transform(new Matrix([[2, 2]]))).toStrictEqual(
      new Matrix([[3, 3]]),
    );
  });

  it('constant column', () => {
    const data = new Matrix([[1, 1, 1, 1, 1, 1, 1, 1]]);

    const scaler = new StandardScaler();
    scaler.fit(data);
    const expected = new Matrix([[0, 0, 0, 0, 0, 0, 0, 0]]);

    expect(scaler.transform(data)).toStrictEqual(expected);
  });
});
