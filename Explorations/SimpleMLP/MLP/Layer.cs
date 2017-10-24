using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Layer
    {
        public int LayerIndex { get; set; }
        public List<HiddenNeuron> Neurons { get; set; }
    }
}
