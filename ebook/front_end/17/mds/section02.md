## angularjs

AngularJS 是一款来自 Google 的前端 JavaScript 框架，大体上讲它属于MVC框架，细致划分的话，它属于MVVM框架，一般用来做单页面应用系统。AngularJS 框架的体积非常小，但是设计理念和功能却非常强大，极大地简化前端开发的负担，它快速成为了 JavaScript 的主流框架，帮助开发者从事 web 开发。

angularjs相关网站：http://www.runoob.com/angularjs/angularjs-tutorial.html


angularjs特性：

1、双向数据绑定
```
<html ng-app="">
......
<p ng-init="name='jack';age=18">
	<input type="text" ng-model="name">
	<input type="text" ng-model="age" >
	我叫：{{name}},我今年{{age}}岁
</p>

```

2、模块
```
<html ng-app="myapp">
......
var app = angular.module('myapp',[]);

```

3、控制器
```
app.controller('uinfo',function($scope){
	$scope.name = 'Tom';
	$scope.arr = [1,2,3];	
})

<div ng-controller = "uinfo">

	<p>我的名字是：{{name}}</p>

	<ul>
		<li ng-repeat="i in arr">{{i}}</li>
	</ul>
</div>

```

4、依赖注入

5、指令

6、过滤器



