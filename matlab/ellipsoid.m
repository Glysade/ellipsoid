
function points = grid_points_in_ellipsoid(A, c, r)
    t = r - c;
    e = zeros(1, size(t, 2));
    for j = 1:size(t, 2)
        e(j) = t(:, j)' * A * t(:, j);
    end
    points = e <= 1;
end

function [volume, exact_volume, points_inside] = ellipsoid_volume_from_grid(A, c, number_of_points)
    d = sqrt(diag(inv(A)));
    Amin = c - d;
    Amax = c + d;
    x_space = (Amax(1) - Amin(1)) / number_of_points;
    y_space = (Amax(2) - Amin(2)) / number_of_points;
    z_space = (Amax(3) - Amin(3)) / number_of_points;
    [x, y, z] = ndgrid(linspace(Amin(1), Amax(1), number_of_points), ...
                       linspace(Amin(2), Amax(2), number_of_points), ...
                       linspace(Amin(3), Amax(3), number_of_points));
    r = [x(:), y(:), z(:)].';
    p = grid_points_in_ellipsoid(A, c, r);
    points_inside = r(:, p >= 1);
    % scatter3(points_inside(1,:), points_inside(2,:), points_inside(3,:));
    volume = sum(p) * (x_space * y_space * z_space);

    e = eig(A);
    exact_volume = (4/3) * pi * prod(1.0 ./ sqrt(e));
    fprintf('Volume from grid: %f, Exact volume: %f\n', volume, exact_volume);

end


function [volume, points_inside] = intersection_volume(A1, c1, A2, c2, number_of_points)
    d1 = sqrt(diag(inv(A1)));
    A1min = c1 - d1;
    A1max = c1 + d1;
    d2 = sqrt(diag(inv(A2)));
    A2min = c2 - d2;
    A2max = c2 + d2;
    Amin = max(A1min, A2min);
    Amax = min(A1max, A2max);
    x_space = (Amax(1) - Amin(1)) / number_of_points;
    y_space = (Amax(2) - Amin(2)) / number_of_points;
    z_space = (Amax(3) - Amin(3)) / number_of_points;
    [x, y, z] = ndgrid(linspace(Amin(1), Amax(1), number_of_points), ...
                       linspace(Amin(2), Amax(2), number_of_points), ...
                       linspace(Amin(3), Amax(3), number_of_points));
    r = [x(:), y(:), z(:)].';
    p1 = grid_points_in_ellipsoid(A1, c1, r);
    p2 = grid_points_in_ellipsoid(A2, c2, r);
    p = p1 & p2;  % Intersection of points inside both ellipsoids
    points_inside = r(:, p >= 1);
    % scatter3(points_inside(1,:), points_inside(2,:), points_inside(3,:));
    volume = sum(p) * (x_space * y_space * z_space);
    fprintf('Intersection volume from grid: %f\n', volume);
end

A1 = [ 1.65932388e-01,  2.94332240e-03, -6.81848476e-03; 2.94332240e-03,  3.13358679e-01,  4.60988976e-05; -6.81848476e-03,  4.60988976e-05,  3.13354109e-01]
c1 = [-2.29916708e-04; -1.92200344e-04; -1.58862120e-05]

A2 = [ 0.07743795,  0.014385  , -0.01813322 ; 0.014385  ,  0.12639254, -0.04517245 ; -0.01813322, -0.04517245,  0.14127035]
c2 = [ 0.49111521;  1.46922175; -1.73698885]

ellipsoid_volume_from_grid(A1, c1, 100);
ellipsoid_volume_from_grid(A2, c2, 100);
intersection_volume(A1, c1, A2, c2, 100);