using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    [Serializable]
    public abstract class Layer
    {
        public List<Neuron> Neurons { get; set; }
        public Layer()
        {
            Neurons = new List<Neuron>();
        }
    }
}
