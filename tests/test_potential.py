import math
import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,"Run in the pinned RFdiffusion PyTorch runtime")
class PotentialTests(unittest.TestCase):
    def setUp(self):
        from baidu_bicycle.potential import bicycle_triangle_v1
        self.potential=bicycle_triangle_v1(pos1=0,pos2=1,pos3=2,binderlen=3,k_dist=5)
        self.xyz=torch.zeros((5,14,3),dtype=torch.float64)
        self.xyz[:3,1]=torch.tensor([[0,0,0],[9,0,0],[4,7,0]],dtype=torch.float64)

    def test_ascent_reduces_penalty(self):
        xyz=self.xyz.requires_grad_()
        before=self.potential.energy(xyz)
        grad=torch.autograd.grad(self.potential.compute(xyz),xyz)[0]
        after=self.potential.energy(xyz+.001*grad)
        self.assertLess(after.item(),before.item())
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(grad[3:].abs().sum().item(),0)

    def test_gradient_finite_difference(self):
        xyz=self.xyz.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(self.potential.compute,(xyz,),eps=1e-6,atol=1e-4))

    def test_translation_invariance(self):
        self.assertAlmostEqual(self.potential.compute(self.xyz).item(),self.potential.compute(self.xyz+200).item(),places=9)

    def test_invalid_coordinates_fail(self):
        self.xyz[0,1,0]=float('nan')
        with self.assertRaises(ValueError): self.potential.compute(self.xyz)

    def test_bounds_fail_instead_of_zero(self):
        with self.assertRaises(ValueError): self.potential.compute(self.xyz[:2])

    def test_target_triangle_zero_penalty(self):
        self.xyz[:3,1]=torch.tensor([[0,0,0],[8,0,0],[4,math.sqrt(48),0]],dtype=torch.float64)
        self.assertLess(self.potential.energy(self.xyz).item(),1e-9)


if __name__ == "__main__": unittest.main()
