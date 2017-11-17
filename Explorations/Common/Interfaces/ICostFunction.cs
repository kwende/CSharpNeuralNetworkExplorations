using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Interfaces
{
    public interface ICostFunction
    {
        double Compute(double target, double output);

        double ComputeDerivativeWRTActivation(double output, double target);
    }
}
