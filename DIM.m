clear; clc; close all;

%% ---------------- PARAMETERS ----------------
AC       = 1;  %coupling strength
As0      = 3.5;  % mean for SHI amplitude
m        = 100;  %number of oscillators 
N        = 50;        % number of graphs
nTrials  = 10;        % simulations per graph

nSweeps  = 3e5;  %gibbs sampling steps
beta     = linspace(0.8,2,nSweeps);   %inverse tempreture
An       = 0.001;  %noise
tstop    = 7;   %simulation time
dt       = 2e-3;  %simulation step length

%% ---------------- RAMP FUNCTION ----------------
a1.k = 10/ tstop;
f1   = @(t, args) t * args.k;

%% ---------------- ER GRAPHS ----------------
p = [0.030 0.2 0.5 0.6 0.8];  %edge density
w_idx = 1;

%% ---------------- HETEROGENEITY ----------------
a = 0;   %As min value
b = 7;    %As max vaue

%% ---------------- PULSE FUNCTION ----------------
Tper = 0.02;  %pulse period
W    = 0.01;   %pulse width
t0   = 10e-3;  %pulse start time 
pulse = @(t) double((t>=t0) & (mod(t-t0,Tper) < W));
%----------------------------
Nsteps   = round(tstop/dt);
idxAll   = 1:m;
s0       = [];

%% ---------------- STORAGE ----------------
MC_all  = zeros(N,nTrials,3);
normVal = zeros(N,nTrials,4);
Q_all   = zeros(N,1);

Nt = Nsteps + 1;

RHO_all = cell(1,3);
CUT_all = cell(1,3);

for mode = 1:3
    RHO_all{mode} = nan(Nt,N,nTrials);
    CUT_all{mode} = nan(Nt,N,nTrials);
end

last_S_mode      = cell(1,3);
last_T1          = [];
last_J           = [];
last_cuts_mode   = cell(1,3);
last_alpha_het   = [];
last_alpha_pulse = [];
R_mode           = cell(1,3);

mode_labels = {'homogeneous','heterogeneous','hetero-pulsed'};

%% ---------------- MAIN LOOP ----------------
for g = 1:N

    %% Generate ER graph
    Jg = zeros(m);
  
    for i = 1:m-1
        for j = i+1:m
            if rand <= p(w_idx)
                Jg(i,j) = -1;
                Jg(j,i) = -1;
            end
        end
    end

    Jg(1:m+1:end) = 0;

    %% Gibbs baseline
    s_gibbs = mcmc_gibbs_spins(Jg, beta, nSweeps, s0);
    Q_gibbs = cut_from_spins(Jg, s_gibbs);
    Q_all(g) = Q_gibbs;

    %% Heterogeneity for this graph
    ks_scale = a + (b-a)*rand(m,1);
    ks_scale = ks_scale / mean(ks_scale);
    As_vec   = As0 * ks_scale;

    for trial = 1:nTrials

        for mode = 1:3

            switch mode
                case 1
                    AS_of_t = @(t) f1(t,a1) * As0 * ones(m,1);

                case 2
                    AS_of_t = @(t) f1(t,a1) * As_vec;

                case 3
                    AS_of_t = @(t) f1(t,a1) * As_vec .* pulse(t);
            end

            %% Simulation
            F1 = @(t,X) dynamics_to_sbm_alpha(X, AC, m, Jg, AS_of_t(t));
            G1 = @(t,X) An * eye(m);

            theta0 = 1*ones(m,1);

            mdl = sde(F1, G1, 'StartState', theta0);

            [S1, T1] = simulate(mdl, Nsteps, 'DeltaTime', dt);

            %% Cut and freezing ratio
            cuts1 = zeros(length(T1),1);
            rmask = false(length(T1),m);

            for k = 1:length(T1)

                x_now = S1(k,:).';
                AS_now = AS_of_t(T1(k));

                for i = 1:m
                    self_i = abs(AS_now(i) * sin(2*pi*x_now(i)));
                    coup_i = abs(AC * Jg(i,:) * sin(pi*(x_now(i) + x_now)));

                    rmask(k,i) = coup_i < self_i;
                end

                spins = sign(S1(k,:));
                spins(spins == 0) = 1;

                ix = find(spins == -1);

                if isempty(ix) || numel(ix) == m
                    cuts1(k) = 0;
                else
                    cuts1(k) = -sum(sum(Jg(ix,setdiff(1:m,ix))));
                end
            end

            MC_all(g,trial,mode) = max(cuts1);

            CUT_all{mode}(1:length(T1),g,trial) = cuts1;
            RHO_all{mode}(1:length(T1),g,trial) = mean(rmask,2);

            %% Save representative last run
            if g == N && trial == nTrials
                last_S_mode{mode}    = S1;
                last_T1              = T1;
                last_J               = Jg;
                last_cuts_mode{mode} = cuts1;
                R_mode{mode}         = rmask;
                last_alpha_het       = As_vec;

                if mode == 3
                    last_alpha_pulse = zeros(length(T1),m);
                    for kk = 1:length(T1)
                        last_alpha_pulse(kk,:) = AS_of_t(T1(kk)).';
                    end
                end
            end
        end

        %% Normalize each graph/trial
        Pm = squeeze(MC_all(g,trial,:)).';

        theta = max([Pm, Q_gibbs]);
        if theta == 0
            theta = 1;
        end

        normVal(g,trial,1:3) = Pm ./ theta;
        normVal(g,trial,4)   = Q_gibbs ./ theta;
    end
end

%% ---------------- FLATTEN RESULTS FOR ALL GRAPHS × TRIALS ----------------
rawHom = reshape(MC_all(:,:,1),[],1);
rawHet = reshape(MC_all(:,:,2),[],1);
rawPul = reshape(MC_all(:,:,3),[],1);

normHom = reshape(normVal(:,:,1),[],1);
normHet = reshape(normVal(:,:,2),[],1);
normPul = reshape(normVal(:,:,3),[],1);

%% ---------------- RESULTS ----------------
disp(['Average raw homogeneous max cut           = ', num2str(mean(rawHom))])
disp(['Average raw heterogeneous max cut         = ', num2str(mean(rawHet))])
disp(['Average raw pulsed heterogeneous max cut  = ', num2str(mean(rawPul))])

disp(['Average normalized homogeneous            = ', num2str(mean(normHom))])
disp(['Average normalized heterogeneous          = ', num2str(mean(normHet))])
disp(['Average normalized pulsed hetero          = ', num2str(mean(normPul))])



%% ---------------- FIGURE 1: wrapped phases ----------------
figure('Name','Wrapped phases','Color','w');

for mode = 1:3

    subplot(3,1,mode);

    Tm = last_T1;
    Sm = last_S_mode{mode};

    if ~isempty(Sm)
        phi = mod(2*pi*Sm,2*pi);
        phi_plot = phi;

        dphi = diff(phi_plot);
        wrapJump = abs(dphi) > pi;
        phi_plot([false(1,size(phi_plot,2)); wrapJump]) = NaN;

        plot(Tm, phi_plot/2, 'LineWidth', 0.8);
        ylim([0 pi]);
        grid on;
        ylabel('phase');
        title(sprintf('Wrapped phases — %s', mode_labels{mode}));
    end
end

xlabel('Time');

%% ---------------- FIGURE 2: raw cut histogram, all graphs × trials ----------------
figure;
hold on;

histogram(rawHom, 'DisplayName','homogeneous', 'FaceAlpha',0.5, 'EdgeColor','none');
histogram(rawHet, 'DisplayName','heterogeneous', 'FaceAlpha',0.5, 'EdgeColor','none');
histogram(rawPul, 'DisplayName','hetero-pulsed', 'FaceAlpha',0.5, 'EdgeColor','none');

xlabel('Raw max cut');
ylabel('Number of runs');
title(sprintf('Raw cut over %d graphs × %d trials', N, nTrials));
legend('Location','best');
grid on;
box on;
hold off;

%% ---------------- FIGURE 3: normalized histogram, all graphs × trials ----------------
edges = linspace(0,1,51);

figure;
hold on;

h1 = histogram(normHom, edges, 'DisplayName','homogeneous');
h2 = histogram(normHet, edges, 'DisplayName','heterogeneous');
h3 = histogram(normPul, edges, 'DisplayName','hetero-pulsed');

set([h1 h2 h3], 'FaceAlpha',0.5, 'EdgeColor','none');

xlabel('Normalized cut');
ylabel('Number of runs');
title(sprintf('Normalized cut over %d graphs × %d trials', N, nTrials));
legend('Location','best');
xlim([0.0 1]);
grid on;
box on;
hold off;

%% ---------------- FIGURE 4: average normalized performance ----------------
figure;

bar_vals = [mean(normHom), mean(normHet), mean(normPul)];
bar_errs = [std(normHom),  std(normHet),  std(normPul)];

bar(bar_vals);
hold on;
errorbar(1:3, bar_vals, bar_errs, '.k', 'LineWidth',1.5);

set(gca,'XTick',1:3, ...
    'XTickLabel',{'homogeneous','heterogeneous','hetero-pulsed'});

ylabel('Normalized score');
title('Average normalized performance over all graphs and trials');
ylim([0 1.1]);
grid on;
box on;
hold off;

% %% ---------------- FIGURE 5: cut vs time for last graph/trial ----------------
% figure;
% hold on;
% 
% plot(last_T1, last_cuts_mode{1}, 'LineWidth',2, 'DisplayName','homogeneous');
% plot(last_T1, last_cuts_mode{2}, 'LineWidth',2, 'DisplayName','heterogeneous');
% plot(last_T1, last_cuts_mode{3}, 'LineWidth',2, 'DisplayName','hetero-pulsed');
% 
% xlabel('Time');
% ylabel('Cut value');
% title('SBM cut vs time, last graph and last trial');
% legend('Location','best');
% grid on;
% box on;





%% ---------------- FIGURE 6: freezing ratio over time ----------------
Time = (0:Nsteps).' * dt;

rho_hom = mean(RHO_all{1}, [2 3], 'omitnan');
rho_het = mean(RHO_all{2}, [2 3], 'omitnan');
rho_pul = mean(RHO_all{3}, [2 3], 'omitnan');

% Smooth over one pulse period
winPulse = round(Tper/dt);

rho_hom_sm = movmean(rho_hom, 50);
rho_het_sm = movmean(rho_het, 50);

% This removes the fast pulse oscillation
rho_pul_avg = movmean(rho_pul, winPulse);

figure;
hold on;

plot(Time, rho_hom_sm, 'LineWidth',2, 'DisplayName','homogeneous');
plot(Time, rho_het_sm, 'LineWidth',2, 'DisplayName','heterogeneous');
plot(Time, rho_pul_avg, 'LineWidth',2, 'DisplayName','hetero-pulsed averaged');

xlabel('Time');
ylabel('Average freezing ratio');
title('Average freezing ratio over all graphs and trials');
ylim([0 1]);
legend('Location','best');
grid on;
box on;

%% ---------------- FIGURE 7: A_S(t) for one random oscillator ----------------
% osc_idx = randi(m);
% 
% figure('Name','A_S(t) for one random oscillator','Color','w');
% 
% for mode = 1:3
% 
%     AS_plot = zeros(length(Time),1);
% 
%     for k = 1:length(Time)
%         tk = Time(k);
% 
%         switch mode
%             case 1
%                 AS_now = f1(tk,a1) * As0 * ones(m,1);
%             case 2
%                 AS_now = f1(tk,a1) * last_alpha_het;
%             case 3
%                 AS_now = f1(tk,a1) * last_alpha_het .* pulse(tk);
%         end
% 
%         AS_plot(k) = AS_now(osc_idx);
%     end
% 
%     subplot(3,1,mode);
%     plot(Time, AS_plot, 'LineWidth',1.5);
%     grid on;
%     ylabel('A_S(t)');
%     title(sprintf('A_S(t) — %s, oscillator %d', mode_labels{mode}, osc_idx));
% end
% 
% xlabel('Time');

%% ---------------- FIGURE 8: mean cut over time ----------------
mean_cut_homog  = mean(CUT_all{1}, [2 3], 'omitnan');
mean_cut_hstat  = mean(CUT_all{2}, [2 3], 'omitnan');
mean_cut_hpulse = mean(CUT_all{3}, [2 3], 'omitnan');

figure;
hold on;
grid on;
box on;

plot(Time, mean_cut_homog,  'LineWidth',1.8, 'DisplayName','homogeneous');
plot(Time, mean_cut_hstat,  'LineWidth',1.8, 'DisplayName','heterogeneous');
plot(Time, mean_cut_hpulse, 'LineWidth',1.8, 'DisplayName','hetero-pulsed');

xlabel('Time');
ylabel('Mean cut across graphs and trials');
title(sprintf('Mean Cut vs Time — p = %.3f', p(w_idx)));
legend('Location','northwest');
xlim([Time(1) Time(end)]);

%% ---------------- FUNCTIONS ----------------
function dxdt = dynamics_to_sbm_alpha(x, AC, n, J, ASvec)

    dxdt = zeros(n,1);

    for i = 1:n
        coup = -AC * J(i,:) * sin(pi*(x(i) + x));
        self = ASvec(i) * sin(2*pi*x(i));

        dxdt(i) = (coup - self) / pi;
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