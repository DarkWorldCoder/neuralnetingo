package loss

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type CategoricalCrossEntropy struct {
}

func (loss *CategoricalCrossEntropy) Calculate(output *mat.Dense, y []uint8) float64 {
	sample_losses := loss.Forward(output, y)
	value := floats.Sum(sample_losses) / float64(len(sample_losses))
	return value

}

func (loss *CategoricalCrossEntropy) Forward(prediction *mat.Dense, target []uint8) []float64 {

	samples := len(target)

	prediction_clipped := mat.DenseCopyOf(prediction)
	prediction_clipped.Apply(func(i, j int, v float64) float64 {
		if v < 1e-7 {
			return 1e-7
		} else if v > 1-1e-7 {
			return 1 - 1e-7
		}
		return v
	}, prediction_clipped)

	sample_losses := make([]float64, samples)
	for i := 0; i < samples; i++ {
		correct_confidence := prediction_clipped.At(i, int(target[i]))
		sample_losses[i] = -math.Log(correct_confidence)
	}

	return sample_losses

}
