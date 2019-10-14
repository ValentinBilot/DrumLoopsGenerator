import torch

def KL_div(means, stds):
	"""
	Works with Batches ie with one extra dimension
	Takes two torch.tensor with means and log stds and computes the KL divergence
	"""
	div = 0
	for i in range(len(means[0])):
		 div+=means[:,i]**2 + stds.exp()[:,i]**2 - 2*stds[:,i] -1
	div = div/2
	return div.mean()

def Gaussian_kernel(x, y):
	"""
	Works with Batches ie with one extra dimension
	Takes two torch.tensor and returns the gaussian kernel for x and y for std=1
	"""
	return (-1*((x-y).norm(2)/2)).exp()

def Inverse_multiquadric_kernel(x, y):
	"""
	Works with Batches ie with one extra dimension
	Takes two torch.tensor and returns the inverse multiquadric kernel for x and y for std=1
	"""
	return 6/(6+(x-y).norm(2))


#Modified to be computationaly useable
def Wasserstein_loss(samples, means, stds, l):
	"""
	Works with Batches ie with one extra dimension
	Takes three torch.tensor and one scalar (l is the regularisation coeff)
	returns the Wasserstein loss for gaussian distribution prior (mu=0, std=1)
	"""
	loss = 0
	loss += KL_div(means, stds)

	n = len(samples) #batch_size
	for l in range(1,n):
		loss+= (l/(n*(n-1))) * Inverse_multiquadric_kernel(samples[0], samples[l])
	return loss

# def Wasserstein_loss(samples, means, stds, l):
# 	"""
# 	Works with Batches ie with one extra dimension
# 	Takes three torch.tensor and one scalar (l is the regularisation coeff)
# 	returns the Wasserstein loss for gaussian distribution prior (mu=0, std=1)
# 	"""
# 	samples, means, stds = samples.cpu(), means.cpu(), stds.cpu()
# 	loss = 0
# 	loss += KL_div(means, stds)
#
# 	n = len(samples) #batch_size
# 	normal_samples = torch.randn(len(samples),len(samples[0]))
# 	for l in range(n):
# 		for j in range(n):
# 			loss-=((2*l)/(n**2))*Gaussian_kernel(normal_samples[l], samples[j])
# 			if l!=j:
# 				loss+= (l/(n*(n-1))) * Gaussian_kernel(normal_samples[l], normal_samples[j])
# 				loss+= (l/(n*(n-1))) * Gaussian_kernel(samples[l], samples[j])
# 	return loss
def compute_kernel(x, y):
	x_size = x.size(0)
	y_size = y.size(0)
	dim = x.size(1)
	x = x.unsqueeze(1) # (x_size, 1, dim)
	y = y.unsqueeze(0) # (1, y_size, dim)
	tiled_x = x.expand(x_size, y_size, dim)
	tiled_y = y.expand(x_size, y_size, dim)
	kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
	#kernel_input = (tiled_x - tiled_y).norm(dim=2)/float(dim)
	return (-kernel_input).exp() # (x_size, y_size)

def compute_mmd(x, y=None):
	"""
	Given to two vectors of size [BxD], compute the maximum mean discrepancy
	between them.
	"""
	if y is None:
		y = torch.randn_like(x)
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	size = float(len(x))
	if size>1:
		scale = size/(size-1)
	else:
		scale=1
	#print(x_kernel.mean().item(), y_kernel.mean().item(), xy_kernel.mean().item())
	#mmd = x_kernel.mean()*size/(size-1) + y_kernel.mean().item()*size/(size-1) - 2*xy_kernel.mean() - 2/size
	mmd = x_kernel.mean()*scale + y_kernel.mean()*scale - 2*xy_kernel.mean()# - (2.0/float(size))
	#print(mmd.item(), size)
	return mmd



if __name__ == "__main__":
	a=torch.randn(512,6)
	div = KL_div(a[:,0:3],a[:,3:6])
	print(div)
	b=torch.randn(512,6)
	k = Gaussian_kernel(a,b)
	print(k)
	sample=torch.randn(512,6)
	loss = Wasserstein_loss(sample, a, b, 0.5)
	print(loss)
