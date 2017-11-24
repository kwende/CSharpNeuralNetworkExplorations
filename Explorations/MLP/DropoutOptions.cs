using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class DropoutOptions
    {
        public List<int> DropoutLayerIndices { get; set; }
        public double ProbabilityOfNeuronDropout { get; set; }

        public DropoutOptions()
        {
            DropoutLayerIndices = new List<int>();
            ProbabilityOfNeuronDropout = 0.0;
        }
    }
}
