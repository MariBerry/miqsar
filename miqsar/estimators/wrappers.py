import numpy as np


class MIWrapper:

    def __init__(self, estimator, pool='mean'):
        self.estimator = estimator
        self.pool = pool

    def apply_pool(self, bags):
        if self.pool == 'mean':
            bags_modified = np.asarray([np.mean(bag, axis=0) for bag in bags])
        elif self.pool == 'extreme':
            bags_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bags_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bags_modified = np.concatenate((bags_max, bags_min), axis=1)
        elif self.pool == 'max':
            bags_modified = np.asarray([np.amax(bag, axis=0) for bag in bags])
        elif self.pool == 'min':
            bags_modified = np.asarray([np.amin(bag, axis=0) for bag in bags])
        return bags_modified

    def fit(self, bags, labels):
        bags_modified = self.apply_pool(bags)
        self.estimator.fit(bags_modified, labels)
        return self.estimator

    def predict(self, bags):
        bags_modified = self.apply_pool(bags)
        preds = self.estimator.predict(bags_modified)
        return preds

    def predict_proba(self, bags):
        bags_modified = self.apply_pool(bags)
        preds = self.estimator.predict_proba(bags_modified)
        return preds


class miWrapper:

    def __init__(self, estimator, pool='mean'):
        self.estimator = estimator
        self.pool = pool

    def apply_pool(self, preds):
        if self.pool == 'mean':
            return np.mean(preds)
        elif self.pool == 'max':
            return np.max(preds)
        elif self.pool == 'min':
            return np.min(preds)
        else:
            print('No exist')
        return preds

    def fit(self, bags, labels):
        bags = np.asarray(bags)
        bags_modified = np.vstack(bags)
        labels_modified = np.hstack([float(lb) * np.array(np.ones(len(bag))) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_modified, labels_modified)
        return self.estimator


class miWrapperRegressor(miWrapper):

    def __init__(self, estimator, pool='mean'):
        super().__init__(estimator=estimator, pool=pool)

    def predict(self, bags):
        preds = [self.apply_pool(self.estimator.predict(bag.reshape(-1, bag.shape[-1]))) for bag in bags]
        return np.asarray(preds)


class miWrapperClassifier(miWrapper):

    def __init__(self, estimator, pool='mean'):
        super().__init__(estimator=estimator, pool=pool)

    def predict(self, bags):
        preds = self.predict_proba(bags)
        preds = np.where(preds > 0.5, 1, 0)
        return preds

    def predict_proba(self, bags):
        preds = [self.apply_pool(self.estimator.predict_proba(bag.reshape(-1, bag.shape[-1]))[:, 1]) for bag in bags]
        return np.asarray(preds)