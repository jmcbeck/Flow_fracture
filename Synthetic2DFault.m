function RoughSurf = Synthetic2DFault(N, H1, H2)

    % from Candela PAGEO appendix
    % create self-affine 2d surface
    % N = size of surface, btw 8 and 11
    % H1 and H2 = Hurst exponents in perpen dirs, pos <1
    % typically 0.6-0.8
    l1 = 1/H1;
    l2 = 1/H2;

    X = (-2*2^N:2:2*2^N)/(2^(N+1));
    X(2^N+1) = 1/2^N;

    Y = (-2*2^N:2:2*2^N)/(2^(N+1));
    Y(2^N+1) = 1/2^N;

    XX = X(ones(1, 2*2^N+1), :);
    YY = Y(ones(1, 2*2^N+1), :)';
    clear X Y

    rho = sqrt(abs(XX).^(2/l1) +  abs(YY).^(2/l2));
    clear XX YY

    phi = rho.^(1 + (l1+l2)/2);
    clear rho

    Z = randn(2*2^N+1, 2*2^N+1);
    W = fftshift(fft2(Z))./phi;
    clear Z
    T = real(ifft2(ifftshift(W)));
    RoughSurf = T-T(2^N+1, 2^N+1);

    
    
end

