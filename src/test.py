import unittest

class TestLineMethods(unittest.TestCase):

    def setUp(self):
        self.widget = Widget('The widget')

    def test_default_widget_size(self):
        self.assertEqual(self.widget.size(), (50,50),
                         'incorrect default size')

    def test_widget_resize(self):
        self.widget.resize(100,150)
        self.assertEqual(self.widget.size(), (100,150),
                         'wrong size after resize')

    def test_curvature(self):
        self.assertEqual('', '')

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def tearDown(self):
        self.widget.dispose()

    @unittest.skip("demonstrating skipping")
    def test_nothing(self):
        self.fail("shouldn't happen")

    @unittest.skipIf(mylib.__version__ < (1, 3),
                     "not supported in this library version")
    def test_format(self):
        # Tests that work for only a certain version of the library.
        pass

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_windows_support(self):
        # windows specific testing code
        pass

    def test_maybe_skipped(self):
        if not external_resource_available():
            self.skipTest("external resource not available")
        # test code that depends on the external resource
        pass
if __name__ == '__main__':
    unittest.main()


@unittest.skip("showing class skipping")
class MySkippedTestCase(unittest.TestCase):
    def test_not_run(self):
        pass

class ExpectedFailureTestCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_fail(self):
        self.assertEqual(1, 0, "broken")


class NumbersTest(unittest.TestCase):

    def test_even(self):
        """
        Test that numbers between 0 and 5 are all even.
        """
        for i in range(0, 6):
            with self.subTest(i=i):
                self.assertEqual(i % 2, 0)

"""
All the assert methods accept a msg argument that, if specified, is used as the error message on failure (see also longMessage). Note that the msg keyword argument can be passed to assertRaises(), assertRaisesRegex(), assertWarns(), assertWarnsRegex() only when they are used as a context manager.

assertEqual(first, second, msg=None)
Test that first and second are equal. If the values do not compare equal, the test will fail.

In addition, if first and second are the exact same type and one of list, tuple, dict, set, frozenset or str or any type that a subclass registers with addTypeEqualityFunc() the type-specific equality function will be called in order to generate a more useful default error message (see also the list of type-specific methods).

Changed in version 3.1: Added the automatic calling of type-specific equality function.

Changed in version 3.2: assertMultiLineEqual() added as the default type equality function for comparing strings.

assertNotEqual(first, second, msg=None)
Test that first and second are not equal. If the values do compare equal, the test will fail.

assertTrue(expr, msg=None)
assertFalse(expr, msg=None)
Test that expr is true (or false).

Note that this is equivalent to bool(expr) is True and not to expr is True (use assertIs(expr, True) for the latter). This method should also be avoided when more specific methods are available (e.g. assertEqual(a, b) instead of assertTrue(a == b)), because they provide a better error message in case of failure.

assertIs(first, second, msg=None)
assertIsNot(first, second, msg=None)
Test that first and second are (or are not) the same object.

New in version 3.1.

assertIsNone(expr, msg=None)
assertIsNotNone(expr, msg=None)
Test that expr is (or is not) None.

New in version 3.1.

assertIn(member, container, msg=None)
assertNotIn(member, container, msg=None)
Test that member is (or is not) in container.

New in version 3.1.

assertIsInstance(obj, cls, msg=None)
assertNotIsInstance(obj, cls, msg=None)
Test that obj is (or is not) an instance of cls (which can be a class or a tuple of classes, as supported by isinstance()). To check for the exact type, use assertIs(type(obj), cls).

New in version 3.2.

assertAlmostEqual(first, second, places=7, msg=None, delta=None)
assertNotAlmostEqual(first, second, places=7, msg=None, delta=None)
Test that first and second are approximately (or not approximately) equal by computing the difference, rounding to the given number of decimal places (default 7), and comparing to zero. Note that these methods round the values to the given number of decimal places (i.e. like the round() function) and not significant digits.

If delta is supplied instead of places then the difference between first and second must be less or equal to (or greater than) delta.

Supplying both delta and places raises a TypeError.

Changed in version 3.2: assertAlmostEqual() automatically considers almost equal objects that compare equal. assertNotAlmostEqual() automatically fails if the objects compare equal. Added the delta keyword argument.

assertGreater(first, second, msg=None)
assertGreaterEqual(first, second, msg=None)
assertLess(first, second, msg=None)
assertLessEqual(first, second, msg=None)
Test that first is respectively >, >=, < or <= than second depending on the method name. If not, the test will fail:

assertRegex(text, regex, msg=None)
assertNotRegex(text, regex, msg=None)
Test that a regex search matches (or does not match) text. In case of failure, the error message will include the pattern and the text (or the pattern and the part of text that unexpectedly matched). regex may be a regular expression object or a string containing a regular expression suitable for use by re.search().

New in version 3.1: Added under the name assertRegexpMatches.

Changed in version 3.2: The method assertRegexpMatches() has been renamed to assertRegex().

New in version 3.2: assertNotRegex().

New in version 3.5: The name assertNotRegexpMatches is a deprecated alias for assertNotRegex().

assertCountEqual(first, second, msg=None)
Test that sequence first contains the same elements as second, regardless of their order. When they don’t, an error message listing the differences between the sequences will be generated.

Duplicate elements are not ignored when comparing first and second. It verifies whether each element has the same count in both sequences. Equivalent to: assertEqual(Counter(list(first)), Counter(list(second))) but works with sequences of unhashable objects as well.

New in version 3.2.

The assertEqual() method dispatches the equality check for objects of the same type to different type-specific methods. These methods are already implemented for most of the built-in types, but it’s also possible to register new methods using addTypeEqualityFunc():

addTypeEqualityFunc(typeobj, function)
Registers a type-specific method called by assertEqual() to check if two objects of exactly the same typeobj (not subclasses) compare equal. function must take two positional arguments and a third msg=None keyword argument just as assertEqual() does. It must raise self.failureException(msg) when inequality between the first two parameters is detected – possibly providing useful information and explaining the inequalities in details in the error message.

New in version 3.1.

The list of type-specific methods automatically used by assertEqual() are summarized in the following table. Note that it’s usually not necessary to invoke these methods directly.

Method

Used to compare

New in

assertMultiLineEqual(a, b)

strings

3.1

assertSequenceEqual(a, b)

sequences

3.1

assertListEqual(a, b)

lists

3.1

assertTupleEqual(a, b)

tuples

3.1

assertSetEqual(a, b)

sets or frozensets

3.1

assertDictEqual(a, b)

dicts

3.1

assertMultiLineEqual(first, second, msg=None)
Test that the multiline string first is equal to the string second. When not equal a diff of the two strings highlighting the differences will be included in the error message. This method is used by default when comparing strings with assertEqual().

New in version 3.1.

assertSequenceEqual(first, second, msg=None, seq_type=None)
Tests that two sequences are equal. If a seq_type is supplied, both first and second must be instances of seq_type or a failure will be raised. If the sequences are different an error message is constructed that shows the difference between the two.

This method is not called directly by assertEqual(), but it’s used to implement assertListEqual() and assertTupleEqual().

New in version 3.1.

assertListEqual(first, second, msg=None)
assertTupleEqual(first, second, msg=None)
Tests that two lists or tuples are equal. If not, an error message is constructed that shows only the differences between the two. An error is also raised if either of the parameters are of the wrong type. These methods are used by default when comparing lists or tuples with assertEqual().

New in version 3.1.

assertSetEqual(first, second, msg=None)
Tests that two sets are equal. If not, an error message is constructed that lists the differences between the sets. This method is used by default when comparing sets or frozensets with assertEqual().

Fails if either of first or second does not have a set.difference() method.

New in version 3.1.

assertDictEqual(first, second, msg=None)
Test that two dictionaries are equal. If not, an error message is constructed that shows the differences in the dictionaries. This method will be used by default to compare dictionaries in calls to assertEqual().

New in version 3.1.

Finally the TestCase provides the following methods and attributes:

fail(msg=None)
Signals a test failure unconditionally, with msg or None for the error message.

failureException
This class attribute gives the exception raised by the test method. If a test framework needs to use a specialized exception, possibly to carry additional information, it must subclass this exception in order to “play fair” with the framework. The initial value of this attribute is AssertionError.

longMessage
This class attribute determines what happens when a custom failure message is passed as the msg argument to an assertXYY call that fails. True is the default value. In this case, the custom message is appended to the end of the standard failure message. When set to False, the custom message replaces the standard message.

The class setting can be overridden in individual test methods by assigning an instance attribute, self.longMessage, to True or False before calling the assert methods.

The class setting gets reset before each test call.

New in version 3.1.

maxDiff
This attribute controls the maximum length of diffs output by assert methods that report diffs on failure. It defaults to 80*8 characters. Assert methods affected by this attribute are assertSequenceEqual() (including all the sequence comparison methods that delegate to it), assertDictEqual() and assertMultiLineEqual().

Setting maxDiff to None means that there is no maximum length of diffs.

New in version 3.2.

Testing frameworks can use the following methods to collect information on the test:

countTestCases()
Return the number of tests represented by this test object. For TestCase instances, this will always be 1.

defaultTestResult()
Return an instance of the test result class that should be used for this test case class (if no other result instance is provided to the run() method).

For TestCase instances, this will always be an instance of TestResult; subclasses of TestCase should override this as necessary.

id()
Return a string identifying the specific test case. This is usually the full name of the test method, including the module and class name.

shortDescription()
Returns a description of the test, or None if no description has been provided. The default implementation of this method returns the first line of the test method’s docstring, if available, or None.

Changed in version 3.1: In 3.1 this was changed to add the test name to the short description even in the presence of a docstring. This caused compatibility issues with unittest extensions and adding the test name was moved to the TextTestResult in Python 3.2.

addCleanup(function, /, *args, **kwargs)
Add a function to be called after tearDown() to cleanup resources used during the test. Functions will be called in reverse order to the order they are added (LIFO). They are called with any arguments and keyword arguments passed into addCleanup() when they are added.

If setUp() fails, meaning that tearDown() is not called, then any cleanup functions added will still be called.

New in version 3.1.

doCleanups()
This method is called unconditionally after tearDown(), or after setUp() if setUp() raises an exception.

It is responsible for calling all the cleanup functions added by addCleanup(). If you need cleanup functions to be called prior to tearDown() then you can call doCleanups() yourself.

doCleanups() pops methods off the stack of cleanup functions one at a time, so it can be called at any time.

New in version 3.1.

classmethod addClassCleanup(function, /, *args, **kwargs)
Add a function to be called after tearDownClass() to cleanup resources used during the test class. Functions will be called in reverse order to the order they are added (LIFO). They are called with any arguments and keyword arguments passed into addClassCleanup() when they are added.

If setUpClass() fails, meaning that tearDownClass() is not called, then any cleanup functions added will still be called.

New in version 3.8.

classmethod doClassCleanups()
This method is called unconditionally after tearDownClass(), or after setUpClass() if setUpClass() raises an exception.

It is responsible for calling all the cleanup functions added by addClassCleanup(). If you need cleanup functions to be called prior to tearDownClass() then you can call doClassCleanups() yourself.

doClassCleanups() pops methods off the stack of cleanup functions one at a time, so it can be called at any time.

New in version 3.8.

class unittest.IsolatedAsyncioTestCase(methodName='runTest')
This class provides an API similar to TestCase and also accepts coroutines as test functions.

New in version 3.8.

coroutine asyncSetUp()
Method called to prepare the test fixture. This is called after setUp(). This is called immediately before calling the test method; other than AssertionError or SkipTest, any exception raised by this method will be considered an error rather than a test failure. The default implementation does nothing.

coroutine asyncTearDown()
Method called immediately after the test method has been called and the result recorded. This is called before tearDown(). This is called even if the test method raised an exception, so the implementation in subclasses may need to be particularly careful about checking internal state. Any exception, other than AssertionError or SkipTest, raised by this method will be considered an additional error rather than a test failure (thus increasing the total number of reported errors). This method will only be called if the asyncSetUp() succeeds, regardless of the outcome of the test method. The default implementation does nothing.

addAsyncCleanup(function, /, *args, **kwargs)
This method accepts a coroutine that can be used as a cleanup function.

run(result=None)
Sets up a new event loop to run the test, collecting the result into the TestResult object passed as result. If result is omitted or None, a temporary result object is created (by calling the defaultTestResult() method) and used. The result object is returned to run()’s caller. At the end of the test all the tasks in the event loop are cancelled.


"""