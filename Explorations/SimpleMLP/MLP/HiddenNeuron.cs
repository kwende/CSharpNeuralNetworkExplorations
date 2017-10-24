using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class HiddenNeuron
    {
        public double Bias { get; private set; }
        public List<Weight> Weights { get; private set; }
    }
}
