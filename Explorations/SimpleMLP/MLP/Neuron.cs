using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public abstract class Neuron
    {
        public Neuron()
        {
            BatchErrors = new List<double>(); 
        }
        public double Output { get; set; }
        public List<double> BatchErrors { get; set; }
    }
}
