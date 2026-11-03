## Independent R survey re-analysis for cross-checking code/svy.py
## Design: svydesign(ids=~PSU, strata=~STR, weights=~w, nest=TRUE); domain via subset(); fasting/subsample weights kept only if >1e-6
suppressMessages({library(survey); library(data.table)})
options(survey.lonely.psu="fail")
B <- "./"
D <- as.data.table(read.csv(gzfile(paste0(B,"data/derived/analytic_for_R.csv.gz")), stringsAsFactors=FALSE))
YRS <- c(E=2,F=2,G=2,H=2,I=2,P=3.2,J=2); MAIN <- c("E","F","G","H","I","P")
mkw <- function(d, wt, cyc){
  Tt <- sum(YRS[cyc]); base <- switch(wt,
    MEC=d$WTMEC, INT=d$WTINT, SAF=d$WTSAF2YR,
    FOL=ifelse(d$cycle=="P", d$WTFOLPRP, d$WTMEC))
  w <- ifelse(is.na(base),0,base) * YRS[d$cycle]/Tt
  ifelse(w>1e-6, w, 0)
}
rcs <- function(x,t){ k<-length(t); p<-function(u) pmax(u,0)^3; out<-cbind(x)
  for(j in 1:(k-2)) out<-cbind(out,(p(x-t[j])-p(x-t[k-1])*(t[k]-t[j])/(t[k]-t[k-1])+p(x-t[k])*(t[k-1]-t[j])/(t[k]-t[k-1]))/(t[k]-t[1])^2)
  colnames(out)<-paste0("S",0:(k-2)); out }
prepdes <- function(cyc, wt, y, tr=NULL){
  d <- D[cycle %in% cyc]
  d$w <- mkw(d, wt, cyc)
  d$y <- d[[y]]; if(!is.null(tr) && tr=="log") d$y <- ifelse(d$y>0, log(d$y), NA)
  d$age <- d$RIDAGEYR-32; d$age2 <- d$age^2
  d$race <- relevel(factor(d$race),"NHW"); d$educ<-relevel(factor(d$educ),">HS"); d$parity<-relevel(factor(d$parity),"0")
  d$smk <- relevel(factor(d$smk),"never"); d$pircat<-relevel(factor(d$pircat),"1.3-3.5"); d$cyc<-factor(d$cycle)
  d$segf <- relevel(factor(ifelse(d$grp=="pp", d$seg, "ctrl")),"ctrl")
  d$dom <- d$grp %in% c("pp","ctrl") & d$w>0 & !is.na(d$y)
  # zero-weight rows: give a placeholder weight so svydesign accepts them; they are then excluded by subset()
  d$wd <- ifelse(d$w>0, d$w, 1)
  des <- svydesign(ids=~PSU, strata=~STR, weights=~wd, nest=TRUE, data=d)
  list(d=d, des=subset(des, dom), full=des)
}
FORM <- ~ segf + age + age2 + race + educ + parity + smk + pircat + married + cyc
res <- list()
add <- function(model, term, b, se, n, df) res[[length(res)+1]] <<- data.frame(model=model, term=term, b=unname(as.numeric(b)), se=unname(as.numeric(se)), n=n, df=df, row.names=NULL)
segfit <- function(tag, cyc, wt, y, tr=NULL, fam=gaussian(), ybin=NULL){
  P <- prepdes(cyc, wt, y, tr); des <- P$des
  if(!is.null(ybin)) des <- update(des, y=as.numeric(ybin(y)))
  f <- svyglm(update(FORM, y ~ .), design=des, family=fam)
  cf <- coef(f); V <- vcov(f)
  for(s in c("0-6","7-12","13-24")){ nm <- paste0("segf",s); add(tag, paste0("seg",s), cf[nm], sqrt(V[nm,nm]), nrow(des$variables), degf(des)) }
  rt <- regTermTest(f, ~segf, method="Wald", df=degf(des))
  add(tag, "wald3_p", rt$p, NA, nrow(des$variables), degf(des))
  cat(tag, ": degf(design)=", degf(des), " residual df(svyglm)=", f$df.residual, "\n")
  invisible(f)
}
segfit("ferritin_log", c("E","F","I","P"), "MEC", "ferritin", "log")
segfit("homa_log", MAIN, "SAF", "homa", "log")
segfit("rbcfol_log", MAIN, "FOL", "rbc_folate", "log")
segfit("bodyiron", c("E","F","I","P"), "MEC", "body_iron")
segfit("shortsleep_logit", MAIN, "INT", "sleep", fam=quasibinomial(), ybin=function(v) v<7)
## RCS ferritin, knots = weighted-domain percentiles 5/35/65/95 of month (same rule as Python: unweighted percentiles among PP with w>0)
P <- prepdes(c("E","F","I","P"), "MEC", "ferritin", "log"); d <- P$d
kn <- quantile(d$month[d$grp=="pp" & d$w>0 & !is.na(d$y)], c(.05,.35,.65,.95), type=7)
cat("R knots:", kn, "\n")
Bm <- rcs(ifelse(is.na(d$month),0,d$month), kn); pp <- as.numeric(d$grp=="pp")
for(j in 1:ncol(Bm)) d[[paste0("S",j-1)]] <- Bm[,j]*pp
d$PP <- pp
des <- subset(svydesign(ids=~PSU, strata=~STR, weights=~wd, nest=TRUE, data=d), dom)
f <- svyglm(y ~ PP + S0 + S1 + S2 + age + age2 + race + educ + parity + smk + pircat + married + cyc, design=des)
G <- rcs(c(3,6,12,18), kn)
for(i in 1:4){ L <- setNames(rep(0,length(coef(f))), names(coef(f))); L["PP"]<-1; L[c("S0","S1","S2")]<-G[i,]
  e <- svycontrast(f, L); add("ferritin_rcs", paste0("m",c(3,6,12,18)[i]), as.numeric(coef(e)), as.numeric(SE(e)), nrow(des$variables), degf(des)) }
## weighted mean ferritin (raw, harmonised), 0-6 mo
P <- prepdes(c("E","F","I","P"), "MEC", "ferritin"); m <- svymean(~y, subset(P$des, segf=="0-6"))
add("ferritin_mean_0-6","mean",coef(m),SE(m),sum(P$d$dom & P$d$segf=="0-6"),NA)
R <- do.call(rbind,res); rownames(R)<-NULL
fwrite(R, paste0(B,"output/rcheck_R.csv")); print(R, digits=6)
