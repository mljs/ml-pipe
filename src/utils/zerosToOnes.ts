export function turnZerosToOnes(vector: number[]): number[] {
  return vector.map((v) => (v === 0 ? 1 : v));
}
