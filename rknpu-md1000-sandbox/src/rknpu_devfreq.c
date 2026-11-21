// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) Rockchip Electronics Co., Ltd.
 * Author: Finley Xiao <finley.xiao@rock-chips.com>
 */

#include <linux/clk.h>
#include <linux/devfreq.h>
#include <linux/of.h>
#include <linux/pm_opp.h>
#include <linux/slab.h>

#include "rknpu_drv.h"
#include "rknpu_devfreq.h"

static int rknpu_clock_target(struct device *dev, unsigned long *freq, u32 flags)
{
	struct rknpu_device *rknpu = dev_get_drvdata(dev);
	struct dev_pm_opp *opp;
	int ret = 0;

	opp = devfreq_recommended_opp(dev, freq, flags);
	if (IS_ERR(opp))
		return PTR_ERR(opp);

	mutex_lock(&rknpu->devfreq_lock);
	if (rknpu->vdd || clk_get_rate(rknpu->devfreq_clk) != *freq)
		ret = dev_pm_opp_set_rate(dev, *freq);
	if (!ret)
		*freq = clk_get_rate(rknpu->devfreq_clk);
	mutex_unlock(&rknpu->devfreq_lock);

	dev_pm_opp_put(opp);
	return ret;
}

static int rknpu_clock_get_rate(struct device *dev, unsigned long *freq)
{
	struct rknpu_device *rknpu = dev_get_drvdata(dev);

	*freq = clk_get_rate(rknpu->devfreq_clk);
	return 0;
}

static int rknpu_clock_get_status(struct device *dev,
				struct devfreq_dev_status *stat)
{
	struct rknpu_device *rknpu = dev_get_drvdata(dev);
	unsigned int cores = rknpu->config->num_irqs;
	unsigned long flags;
	ktime_t busy = 0;
	unsigned int i;

	spin_lock_irqsave(&rknpu->irq_lock, flags);
	for (i = 0; i < cores; i++)
		busy += rknpu->subcore_datas[i].timer.total_busy_time;
	spin_unlock_irqrestore(&rknpu->irq_lock, flags);

	stat->current_frequency = clk_get_rate(rknpu->devfreq_clk);
	stat->total_time = RKNPU_LOAD_INTERVAL / NSEC_PER_USEC;
	stat->busy_time = min_t(unsigned long, ktime_to_us(busy) / cores,
				stat->total_time);
	return 0;
}

void rknpu_devfreq_lock(struct rknpu_device *rknpu)
{
	mutex_lock(&rknpu->devfreq_lock);
}

void rknpu_devfreq_unlock(struct rknpu_device *rknpu)
{
	mutex_unlock(&rknpu->devfreq_lock);
}

int rknpu_devfreq_init(struct rknpu_device *rknpu)
{
	static const char * const regulators[] = { "rknpu", NULL };
	struct device *dev = rknpu->dev;
	struct devfreq_dev_profile *profile;
	struct dev_pm_opp *opp;
	struct clk *clk;
	unsigned long initial_rate, rate = 0;
	int ret;

	if (!of_machine_is_compatible("rockchip,rk3566-md1000"))
		return 0;

	/* Track the CRU clock selected by the MD1000 DT. */
	clk = devm_clk_get(dev, "clk");
	if (IS_ERR(clk))
		return PTR_ERR(clk);
	initial_rate = min(clk_get_rate(clk), 600000000UL);

	ret = devm_pm_opp_set_clkname(dev, "clk");
	if (ret)
		return ret;
	if (rknpu->vdd) {
		ret = devm_pm_opp_set_regulators(dev, regulators);
		if (ret)
			return ret;
	}
	ret = devm_pm_opp_of_add_table(dev);
	if (ret)
		return ret;

	while (!IS_ERR(opp = dev_pm_opp_find_freq_ceil(dev, &rate))) {
		dev_pm_opp_put(opp);
		if (rate > initial_rate || clk_round_rate(clk, rate) != rate) {
			ret = dev_pm_opp_disable(dev, rate);
			if (ret)
				return ret;
		}
		rate++;
	}
	rate = initial_rate;
	opp = dev_pm_opp_find_freq_exact(dev, rate, true);
	if (IS_ERR(opp))
		return PTR_ERR(opp);
	dev_pm_opp_put(opp);

	if (dev_pm_opp_get_opp_count(dev) < 2)
		return -ERANGE;

	if (rknpu->vdd) {
		/* Apply the initial OPP even when the clock already matches. */
		ret = dev_pm_opp_set_rate(dev, initial_rate);
		if (ret)
			return ret;
	}

	profile = devm_kzalloc(dev, sizeof(*profile), GFP_KERNEL);
	if (!profile)
		return -ENOMEM;

	profile->initial_freq = initial_rate;
	profile->polling_ms = 1000;
	profile->target = rknpu_clock_target;
	profile->get_cur_freq = rknpu_clock_get_rate;
	profile->get_dev_status = rknpu_clock_get_status;

	rknpu->devfreq_clk = clk;
	rknpu->devfreq = devm_devfreq_add_device(dev, profile, "performance", NULL);
	if (IS_ERR(rknpu->devfreq)) {
		ret = PTR_ERR(rknpu->devfreq);
		rknpu->devfreq = NULL;
		rknpu->devfreq_clk = NULL;
		return ret;
	}

	dev_info(dev, "MD1000 %s devfreq enabled (boot rate %lu Hz)\n",
		 rknpu->vdd ? "voltage and clock" : "clock-only", initial_rate);
	return 0;
}

void rknpu_devfreq_remove(struct rknpu_device *rknpu)
{
	if (rknpu->devfreq) {
		devm_devfreq_remove_device(rknpu->dev, rknpu->devfreq);
		rknpu->devfreq = NULL;
	}
}

int rknpu_devfreq_runtime_suspend(struct device *dev)
{
	return 0;
}

int rknpu_devfreq_runtime_resume(struct device *dev)
{
	return 0;
}
