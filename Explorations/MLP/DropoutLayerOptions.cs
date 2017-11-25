using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class DropoutLayerOptions
    {
        public double ProbabilityOfDropout { get; private set; }
        public int[] DropoutLayerIndices { get; private set; }

        public DropoutLayerOptions(double probabilityOfDropout, params int[] layerIndices)
        {
            DropoutLayerIndices = layerIndices;
            ProbabilityOfDropout = probabilityOfDropout;
        }
    }
}
