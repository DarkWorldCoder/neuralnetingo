package activations

import (
	"math"
	"slices"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Activation_Softmax struct {
	Output mat.Dense
}

func (a *Activation_Softmax) Forward(inputs *mat.Dense) {
	rows, cols := inputs.Dims()
	a.Output = *mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		row := softmax(inputs.RawRowView(i))
		a.Output.SetRow(i, row)
	}

}

func softmax(input []float64) []float64 {
	output := make([]float64, len(input))
	expValues := make([]float64, len(input))

	maxValue := slices.Max(input)

	for i := range expValues {
		expValues[i] = math.Exp(input[i] - maxValue)
	}

	sum := floats.Sum(expValues)

	for i := range output {
		output[i] = expValues[i] / sum
	}

	return output
}
