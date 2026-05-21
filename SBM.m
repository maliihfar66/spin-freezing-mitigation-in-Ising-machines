clc;
clear all;
close all;

%% ---------------- PARAMETERS ----------------
nOsc    = 100;%number of Oscillators
Ngraphs = 50;%number of graphs 
nTrials = 10;      % NEW: number of simulations per graph
p       = 0.03;%graph density

eps   = 0.05;%epsilon value

An    = 0.001;%noise
tstop = 30;%simulation time 
tstep = 2e-3;% simulation step length


% ----- heterogeneity range for alpha -----
a = 0.1; %min of alpha
b = 2;%max of alpha

% ----- pulse parameters for alpha -----
Tper = 0.02;%pulse period
W    = 0.01;%pulse width
t0   = 0.0;%pulse start time
pulse = @(t) double((t >= t0) & (mod(t - t0, Tper) < W));

% ----- Gibbs sampling parameters -----
nSweeps = 3e5;%gibbs sampling steps
beta    = linspace(0.8,2,nSweeps);%inverse temperature
s0      = [];
%---------
delta = 1;
Ke    = 1;
a1.k = 3/ tstop;
f1   = @(t, args) t * args.k;

%% ---------------- STORAGE ----------------
MC_all      = zeros(Ngraphs, nTrials, 3);
Q_gibbs_all = zeros(Ngraphs,1);
normVal     = zeros(Ngraphs, nTrials, 4);

last_S_mode      = cell(1,3);
last_T1          = [];
last_J           = [];
last_cuts_mode   = cell(1,3);
last_alpha_het   = [];
last_alpha_pulse = [];

Nt = round(tstop/tstep) + 1;

RHO_all = cell(1,3);
CUT_all = cell(1,3);

for mode = 1:3
    RHO_all{mode} = nan(Nt, Ngraphs, nTrials);
    CUT_all{mode} = nan(Nt, Ngraphs, nTrials);
end

R_mode = cell(1,3);
mode_labels = {'homogeneous','heterogeneous','hetero-pulsed'};

%% ---------------- MAIN LOOP OVER ER GRAPHS ----------------
for g = 1:Ngraphs

    %% ----- generate ER graph -----
    J = zeros(nOsc);

    for i = 1:nOsc-1
        for j = i+1:nOsc
            if rand <= p
                J(i,j) = -1;
                J(j,i) = -1;
            end
        end
    end

    J(1:nOsc+1:end) = 0;

    %% ----- Gibbs baseline for this graph -----
    s_gibbs = mcmc_gibbs_spins(J, beta, nSweeps, s0);
    Q_gibbs = cut_from_spins(J, s_gibbs);
    Q_gibbs_all(g) = Q_gibbs;

    %% ----- heterogeneous alpha for this graph -----
    alpha_vec = a + (b-a) * rand(nOsc,1);
    alpha_vec = alpha_vec / mean(alpha_vec);

    %% ----- repeat simulations nTrials times -----
    for trial = 1:nTrials

        MC_trial = zeros(1,3);

        for mode = 1:3

            switch mode
                case 1
                    alphafun = @(t) ones(nOsc,1);

                case 2
                    alphafun = @(t) alpha_vec;

                case 3
                    alphafun = @(t) alpha_vec * pulse(t);
            end

            %% ----- SBM simulation -----
            F1 = @(t,X) dynamics_to_sbm_alpha(X, f1(t,a1), eps, nOsc, J, delta, Ke, alphafun(t));
            G1 = @(t,X) An * eye(nOsc);

            % NEW: random initial condition for each trial
            X0 = 0.1 * randn(nOsc,1);

            obj1 = sde(F1, G1, 'StartState', X0);

            [S1, T1] = simulate(obj1, round(tstop/tstep), ...
                'DeltaTime', tstep);

            %% ----- compute cut over time + freezing mask -----
            cuts1 = zeros(length(T1),1);
            rmask = false(length(T1), nOsc);

            for k = 1:length(T1)

                x_now = S1(k,:).';
                P_now = f1(T1(k), a1);
                alpha_now = alphafun(T1(k));

                for i = 1:nOsc
                    lin_i   = alpha_now(i) * (-delta + P_now) * x_now(i);
                    cub_i   = -alpha_now(i) * Ke * x_now(i)^3;
                    coup_i  = eps * (J(i,:) * x_now);
                    other_i = lin_i + cub_i;

                    rmask(k,i) = (abs(other_i) > abs(coup_i)) && ((other_i * coup_i) < 0);
                end

                spins = sign(S1(k,:));
                spins(spins == 0) = 1;

                ix = find(spins == -1);

                if isempty(ix) || numel(ix) == nOsc
                    cuts1(k) = 0;
                else
                    cuts1(k) = -sum(sum(J(ix, setdiff(1:nOsc, ix))));
                end
            end

            MC_trial(mode) = max(cuts1);
            MC_all(g,trial,mode) = MC_trial(mode);

            CUT_all{mode}(1:length(T1), g, trial) = cuts1;
            RHO_all{mode}(1:length(T1), g, trial) = mean(rmask, 2);

            if g == Ngraphs && trial == nTrials
                last_S_mode{mode}    = S1;
                last_T1              = T1;
                last_J               = J;
                last_cuts_mode{mode} = cuts1;
                R_mode{mode}         = rmask;
                last_alpha_het       = alpha_vec;

                if mode == 3
                    last_alpha_pulse = zeros(length(T1), nOsc);
                    for kk = 1:length(T1)
                        last_alpha_pulse(kk,:) = alphafun(T1(kk)).';
                    end
                end
            end
        end

        %% ----- normalize per graph and trial -----
        theta = max([MC_trial(1), MC_trial(2), MC_trial(3), Q_gibbs]);

        if theta == 0
            theta = 1;
        end

        normVal(g,trial,1) = MC_trial(1) / theta;
        normVal(g,trial,2) = MC_trial(2) / theta;
        normVal(g,trial,3) = MC_trial(3) / theta;
        normVal(g,trial,4) = Q_gibbs   / theta;
    end
end

%% ---------------- FLATTEN NORMALIZED VALUES ----------------
normHom = reshape(normVal(:,:,1), [], 1);
normHet = reshape(normVal(:,:,2), [], 1);
normPul = reshape(normVal(:,:,3), [], 1);

%% ---------------- RESULTS ----------------
disp(['Average raw homogeneous max cut           = ', num2str(mean(MC_all(:,:,1), 'all'))])
disp(['Average raw heterogeneous max cut         = ', num2str(mean(MC_all(:,:,2), 'all'))])
disp(['Average raw pulsed heterogeneous max cut  = ', num2str(mean(MC_all(:,:,3), 'all'))])

disp(['Average normalized homogeneous            = ', num2str(mean(normHom))])
disp(['Average normalized heterogeneous          = ', num2str(mean(normHet))])
disp(['Average normalized pulsed hetero          = ', num2str(mean(normPul))])




%% ---------------- FIGURE 1: amplitudes vs time pulsed ----------------
figure;
plot(last_T1, last_S_mode{1}, 'LineWidth', 1.2);
xlabel('Time');
ylabel('x');
title('Homogeneous \alpha: amplitudes vs time');
grid on;
box on;

%% ---------------- FIGURE 2: raw cut histogram all trials ----------------
figure;
hold on;

histogram(reshape(MC_all(:,:,1), [], 1), 'DisplayName','homogeneous', 'FaceAlpha',0.5, 'EdgeColor','none');
histogram(reshape(MC_all(:,:,2), [], 1), 'DisplayName','heterogeneous', 'FaceAlpha',0.5, 'EdgeColor','none');
histogram(reshape(MC_all(:,:,3), [], 1), 'DisplayName','hetero-pulsed', 'FaceAlpha',0.5, 'EdgeColor','none');

xlabel('Raw max cut');
ylabel('Count');
title(sprintf('Raw max cut over %d graphs × %d trials', Ngraphs, nTrials));
legend('Location','best');
grid on;
box on;
hold off;

%% ---------------- FIGURE 3: normalized cut histogram all trials ----------------
edges = linspace(0,1,51);

figure;
hold on;

histogram(normHom, edges, 'DisplayName','homogeneous', 'FaceAlpha',0.5, 'EdgeColor','none');
histogram(normHet, edges, 'DisplayName','heterogeneous', 'FaceAlpha',0.5, 'EdgeColor','none');
histogram(normPul, edges, 'DisplayName','hetero-pulsed', 'FaceAlpha',0.5, 'EdgeColor','none');

xlabel('Normalized cut');
ylabel('Count');
title(sprintf('Normalized cut over %d graphs × %d trials', Ngraphs, nTrials));
legend('Location','best');
xlim([0.7 1]);
grid on;
box on;
hold off;

%% ---------------- FIGURE 4: average normalized performance ----------------
figure;

bar_vals = [mean(normHom), mean(normHet), mean(normPul)];
bar_errs = [std(normHom),  std(normHet),  std(normPul)];

bar(bar_vals);
hold on;
errorbar(1:3, bar_vals, bar_errs, '.k', 'LineWidth', 1.5);

set(gca, 'XTick', 1:3, ...
    'XTickLabel', {'homogeneous','heterogeneous','hetero-pulsed'});

ylabel('Normalized score');
title('Average normalized performance over all trials');
ylim([0 1.1]);
grid on;
box on;
hold off;

%% ---------------- FIGURE 5: cut vs time for last run ----------------
figure;
hold on;

plot(last_T1, last_cuts_mode{1}, 'LineWidth', 2, 'DisplayName','homogeneous');
plot(last_T1, last_cuts_mode{2}, 'LineWidth', 2, 'DisplayName','heterogeneous');
plot(last_T1, last_cuts_mode{3}, 'LineWidth', 2, 'DisplayName','hetero-pulsed');

xlabel('Time');
ylabel('Cut value');
title('SBM cut vs time, last graph and last trial');
legend('Location','best');
grid on;
box on;





%% ---------------- FIGURE 6: freezing ratio over time ----------------
figure;
hold on;

Time = (0:Nt-1).' * tstep;

rho_hom = mean(RHO_all{1}, [2 3], 'omitnan');
rho_het = mean(RHO_all{2}, [2 3], 'omitnan');
rho_pul = mean(RHO_all{3}, [2 3], 'omitnan');

win = 50;

rho_hom_sm = movmean(rho_hom, win);
rho_het_sm = movmean(rho_het, win);
rho_pul_sm = movmean(rho_pul, win);

plot(Time, rho_hom_sm, 'LineWidth', 2, 'DisplayName','homogeneous');
plot(Time, rho_het_sm, 'LineWidth', 2, 'DisplayName','heterogeneous');
plot(Time, rho_pul_sm, 'LineWidth', 2, 'DisplayName','hetero-pulsed');

xlabel('Time');
ylabel('Average freezing ratio');
title(sprintf('Average freezing ratio over all graphs/trials, window = %d', win));
ylim([0 1]);
legend('Location','best');
grid on;
box on;

%% ---------------- FIGURE 7: heatmap homogeneous ----------------
figure;
M1 = double(R_mode{1}.');
h12 = heatmap(M1);
h12.GridVisible = 'off';
h12.Colormap = cool;
h12.ColorLimits = [0 1];
h12.XDisplayLabels = repmat("", 1, size(M1,2));
h12.YDisplayLabels = string(1:nOsc);
title('Frozen status over time — homogeneous');
xlabel('Time step');
ylabel('Oscillator');

%% ---------------- FIGURE 8: heatmap heterogeneous ----------------
figure;
M2 = double(R_mode{2}.');
h13 = heatmap(M2);
h13.GridVisible = 'off';
h13.Colormap = cool;
h13.ColorLimits = [0 1];
h13.XDisplayLabels = repmat("", 1, size(M2,2));
h13.YDisplayLabels = string(1:nOsc);
title('Frozen status over time — heterogeneous');
xlabel('Time step');
ylabel('Oscillator');

%% ---------------- FIGURE 9: heatmap hetero-pulsed ----------------
figure;
M3 = double(R_mode{3}.');
h14 = heatmap(M3);
h14.GridVisible = 'off';
h14.Colormap = cool;
h14.ColorLimits = [0 1];
h14.XDisplayLabels = repmat("", 1, size(M3,2));
h14.YDisplayLabels = string(1:nOsc);
title('Frozen status over time — hetero-pulsed');
xlabel('Time step');
ylabel('Oscillator');

%% ---------------- FIGURE 10: mean cut over time ----------------
figure;
hold on;

mean_cut_hom = mean(CUT_all{1}, [2 3], 'omitnan');
mean_cut_het = mean(CUT_all{2}, [2 3], 'omitnan');
mean_cut_pul = mean(CUT_all{3}, [2 3], 'omitnan');

win_cut = 50;

mean_cut_hom_sm = movmean(mean_cut_hom, win_cut);
mean_cut_het_sm = movmean(mean_cut_het, win_cut);
mean_cut_pul_sm = movmean(mean_cut_pul, win_cut);

plot(Time, mean_cut_hom_sm, 'LineWidth', 2, 'DisplayName','homogeneous');
plot(Time, mean_cut_het_sm, 'LineWidth', 2, 'DisplayName','heterogeneous');
plot(Time, mean_cut_pul_sm, 'LineWidth', 2, 'DisplayName','hetero-pulsed');

xlabel('Time');
ylabel('Mean cut across graphs and trials');
title(sprintf('Mean cut over time, window = %d', win_cut));
legend('Location','best');
grid on;
box on;

%% ---------------- FUNCTIONS ----------------
function dxdt = dynamics_to_sbm_alpha(x, P, eps, n, J, delta, Ke, alpha_vec)

    dxdt = zeros(n,1);

    for i = 1:n
        sumJx = J(i,:) * x;

        lin  = alpha_vec(i) * (-delta + P) * x(i);
        cub  = -alpha_vec(i) * Ke * x(i)^3;
        coup = eps * sumJx;

        dxdt(i) = lin + cub + coup;
    end
end

function s = mcmc_gibbs_spins(J, beta, nSweeps, s0)

    n = size(J,1);

    if nargin < 4 || isempty(s0)
        s = sign(randn(n,1));
        s(s==0) = 1;
    else
        s = s0(:);
        s(s==0) = 1;
    end

    for sweep = 1:nSweeps
        idx = randperm(n);

        for i = idx
            h_i = J(i,:) * s;

            if beta(sweep) <= 0
                pspin = 0.5;
            else
                pspin = 1 ./ (1 + exp(-2 * beta(sweep) * h_i));
            end

            s(i) = 2 * (rand < pspin) - 1;
        end
    end
end

function cutVal = cut_from_spins(J, s)

    A = (J == -1);

    cutVal = 0.5 * sum(sum(triu(A,1) .* (1 - (s*s.'))));
end