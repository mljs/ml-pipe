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

  it('invese transform', () => {
    const data = new Matrix([
      [0, 1],
      [1, 2],
      [4356, 1],
      [1, 345],
    ]);

    const scaler = new StandardScaler();
    const transformed = scaler.fitTransform(data);
    expect(scaler.inverseTransform(transformed)).toStrictEqual(data);
  });

  it('test throw if not fitted', () => {
    const data = new Matrix([
      [0, 1],
      [1, 2],
      [4356, 1],
      [1, 345],
    ]);

    const scaler = new StandardScaler();

    expect(() => scaler.transform(data)).toThrow(
      'You must fit the transformer before using it',
    );
  });
});
