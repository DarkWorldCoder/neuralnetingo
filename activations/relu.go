package activations

import "gonum.org/v1/gonum/mat"

type Activation_ReLU struct {
	Output mat.Dense
}

func (a *Activation_ReLU) Forward(input *mat.Dense) {
	r, c := input.Dims()
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if input.At(i, j) > 0 {
				result.Set(i, j, input.At(i, j))
			} else {
				result.Set(i, j, 0)
			}
		}
	}
	a.Output = *result
}
