package dataset

import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func PlotScatterData(values [][]float64, class_count int) {
	x, y := make([]float64, len(values)), make([]float64, len(values))
	for i := range values {
		x[i] = values[i][0]
		y[i] = values[i][1]
	}

	p := plot.New()
	p.Title.Text = "Spiral Data"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	for class := 0; class < class_count; class++ {
		offset := len(values) / class_count
		pts := make(plotter.XYs, 0)
		for i := 0; i < offset; i++ {
			index := class*offset + i
			pts = append(pts, plotter.XY{X: x[index], Y: y[index]})
		}
		s, err := plotter.NewScatter(pts)
		if err != nil {
			panic(err)
		}
		s.GlyphStyle.Color = color.RGBA{R: uint8(rand.Intn(255)), G: uint8(rand.Intn(255)), B: uint8(rand.Intn(255)), A: 255}
		p.Add(s)
		p.Legend.Add(fmt.Sprintf("Class %d", class), s)
	}

	if err := p.Save(6*vg.Inch, 6*vg.Inch, "spiral_data.png"); err != nil {
		panic(err)
	}
}
